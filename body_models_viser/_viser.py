from __future__ import annotations

import base64
import dataclasses
from importlib.resources import files
from typing import Any

import numpy as np
import numpy.typing as npt
from body_models.smpl.numpy import SMPL
from jaxtyping import Float
from viser import _messages

from ._client_autobuild import ensure_client_is_built


@dataclasses.dataclass
class BodyModelsViserSmplMessage(_messages.Message, include_in_scene_serialization=True):
    name: str
    vertex_count: int
    face_count: int
    lbs_weights: npt.NDArray[np.float32]
    faces: npt.NDArray[np.uint32]
    rest_joints: npt.NDArray[np.float32]
    rest_vertices: npt.NDArray[np.float32]
    joint_transforms: npt.NDArray[np.float32]
    pose_offsets: npt.NDArray[np.float32]
    global_rotation: npt.NDArray[np.float32]
    global_translation: npt.NDArray[np.float32]
    props: dict[str, Any]


@dataclasses.dataclass
class BodyModelsViserPoseMessage(_messages.Message, include_in_scene_serialization=True):
    name: str
    rest_joints: npt.NDArray[np.float32] | None
    rest_vertices: npt.NDArray[np.float32] | None
    joint_transforms: npt.NDArray[np.float32] | None
    pose_offsets: npt.NDArray[np.float32] | None
    global_rotation: npt.NDArray[np.float32]
    global_translation: npt.NDArray[np.float32]


class SmplBodyHandle:
    def __init__(
        self,
        scene: Any,
        name: str,
        model: SMPL,
        pose: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
    ) -> None:
        self.scene = scene
        self.name = name
        self.model = model
        self.pose = pose

    @property
    def shape(self) -> Float[np.ndarray, "10"]:
        return self.pose["shape"]

    @shape.setter
    def shape(self, value: Float[np.ndarray, "10"]) -> None:
        self.set_pose(shape=value)

    @property
    def body_pose(self) -> Float[np.ndarray, "23 3"]:
        return self.pose["body_pose"]

    @body_pose.setter
    def body_pose(self, value: Float[np.ndarray, "23 3"]) -> None:
        self.set_pose(body_pose=value)

    @property
    def pelvis_rotation(self) -> Float[np.ndarray, "3"]:
        return self.pose["pelvis_rotation"]

    @pelvis_rotation.setter
    def pelvis_rotation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_pose(pelvis_rotation=value)

    @property
    def global_rotation(self) -> Float[np.ndarray, "3"]:
        return self.pose["global_rotation"]

    @global_rotation.setter
    def global_rotation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_pose(global_rotation=value)

    @property
    def global_translation(self) -> Float[np.ndarray, "3"]:
        return self.pose["global_translation"]

    @global_translation.setter
    def global_translation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_pose(global_translation=value)

    def set_pose(self, **params: Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]) -> None:
        shape_changed = "shape" in params
        pose_changed = bool({"shape", "body_pose", "pelvis_rotation"} & params.keys())
        for key, value in params.items():
            self.pose[key] = np.asarray(value, dtype=np.float32).copy()
        identity = self.model.prepare_identity(self.pose["shape"]) if shape_changed else None
        prepared_pose = _prepare_pose(self.model, self.pose, identity) if pose_changed else None
        _queue_connected_clients(self.scene, _pose_message(self.name, self.pose, identity, prepared_pose))

    def remove(self) -> None:
        self.scene._websock_interface.queue_message(_messages.RemoveSceneNodeMessage(self.name))


def add_body_model(
    scene: Any,
    name: str,
    model: SMPL,
    *,
    color: tuple[int, int, int] = (180, 180, 180),
    wireframe: bool = False,
    opacity: float | None = None,
    flat_shading: bool = False,
    side: str = "front",
    material: str = "standard",
    scale: float | tuple[float, float, float] = 1.0,
    cast_shadow: bool = True,
    receive_shadow: bool | float = True,
) -> SmplBodyHandle:
    if not isinstance(model, SMPL):
        raise TypeError("body-models-viser currently supports SMPL only.")
    if model.rotation_type != "axis_angle":
        raise ValueError("body-models-viser currently supports SMPL axis-angle parameters only.")

    pose = {key: np.asarray(value, dtype=np.float32).copy() for key, value in model.get_rest_pose().items()}
    identity = model.prepare_identity(pose["shape"])
    prepared_pose = _prepare_pose(model, pose, identity)
    props = {
        "color": color,
        "wireframe": wireframe,
        "opacity": opacity,
        "flat_shading": flat_shading,
        "side": side,
        "material": material,
        "scale": scale,
        "cast_shadow": cast_shadow,
        "receive_shadow": receive_shadow,
    }
    message = BodyModelsViserSmplMessage(
        name=name,
        vertex_count=int(model.weights.v_template.shape[0]),
        face_count=int(model.faces.shape[0]),
        lbs_weights=_array(model.weights.lbs_weights, np.float32),
        faces=_array(model.faces, np.uint32),
        rest_joints=_array(identity["rest_joints"], np.float32),
        rest_vertices=_array(identity["rest_vertices"], np.float32),
        joint_transforms=_array(prepared_pose["joint_transforms"], np.float32),
        pose_offsets=_array(prepared_pose["pose_offsets"], np.float32),
        global_rotation=_array(pose["global_rotation"], np.float32),
        global_translation=_array(pose["global_translation"], np.float32),
        props=props,
    )
    _queue_connected_clients(scene, _messages.RunJavascriptMessage(_install_javascript()), flush=True)
    _queue_connected_clients(scene, message, after_flush=True)

    @scene._websock_interface.on_client_connect
    def _(client: Any) -> None:
        _install_client_runtime(client)
        _queue_after_flush(client.get_message_buffer(), message)

    return SmplBodyHandle(scene, name, model, pose)


def _install_client_runtime(client: Any) -> None:
    client.queue_message(_messages.RunJavascriptMessage(_install_javascript()))
    client.get_message_buffer().flush()


def _queue_connected_clients(
    scene: Any,
    message: _messages.Message,
    *,
    flush: bool = False,
    after_flush: bool = False,
) -> None:
    for state in scene._websock_interface._client_state_from_id.values():
        if after_flush:
            _queue_after_flush(state.message_buffer, message)
        else:
            state.message_buffer.push(message)
            if flush:
                state.message_buffer.flush()


def _queue_after_flush(buffer: Any, message: _messages.Message) -> None:
    def queue_message() -> None:
        buffer.event_loop.call_later(0.05, buffer.push, message)

    buffer.event_loop.call_soon_threadsafe(queue_message)


def _install_javascript() -> str:
    ensure_client_is_built()
    source = (files(__package__) / "client" / "body-models-viser.js").read_text()
    wasm = base64.b64encode(_wasm_bytes()).decode("ascii")
    return f"if (!window.BodyModelsViser) {{\n{source}\nwindow.BodyModelsViser = BodyModelsViser;\nwindow.BodyModelsViser.install('{wasm}');\n}}"


def _wasm_bytes() -> bytes:
    ensure_client_is_built()
    return (files(__package__) / "client" / "body-models-viser.wasm").read_bytes()


def _pose_message(
    name: str,
    pose: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
    identity: dict[str, Any] | None,
    prepared_pose: dict[str, Any] | None,
) -> BodyModelsViserPoseMessage:
    return BodyModelsViserPoseMessage(
        name=name,
        rest_joints=None if identity is None else _array(identity["rest_joints"], np.float32),
        rest_vertices=None if identity is None else _array(identity["rest_vertices"], np.float32),
        joint_transforms=None if prepared_pose is None else _array(prepared_pose["joint_transforms"], np.float32),
        pose_offsets=None if prepared_pose is None else _array(prepared_pose["pose_offsets"], np.float32),
        global_rotation=_array(pose["global_rotation"], np.float32),
        global_translation=_array(pose["global_translation"], np.float32),
    )


def _prepare_pose(
    model: SMPL,
    params: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
    identity: dict[str, Any] | None,
) -> dict[str, Any]:
    identity = model.prepare_identity(params["shape"]) if identity is None else identity
    return model.prepare_pose(
        params["body_pose"],
        params["pelvis_rotation"],
        identity=identity,
    )


def _array(array: Any, dtype: Any) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<"))
