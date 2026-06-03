from __future__ import annotations

import base64
import dataclasses
import inspect
from collections.abc import Callable, Iterable
from importlib.resources import files
from typing import Any

import numpy as np
import numpy.typing as npt
from body_models.anny.numpy import ANNY
from body_models.flame.numpy import FLAME
from body_models.garment_measurements.numpy import GarmentMeasurements
from body_models.garment_measurements.pose import pack_pose
from body_models.mano.numpy import MANO
from body_models.mhr.numpy import MHR
from body_models.skel.numpy import SKEL
from body_models.smpl.numpy import SMPL
from body_models.smplh.numpy import SMPLH
from body_models.smplx.numpy import SMPLX
from body_models.soma.numpy import SOMA
from jaxtyping import Float
from viser import _messages

from ._client_autobuild import ensure_client_is_built


@dataclasses.dataclass
class BodyModelsViserModelMessage(_messages.Message, include_in_scene_serialization=True):
    name: str
    vertex_count: int
    lbs_weights: npt.NDArray[np.float32]
    faces: npt.NDArray[np.uint32]
    rest_vertices: npt.NDArray[np.float32]
    skinning_transforms: npt.NDArray[np.float32]
    pose_offsets: npt.NDArray[np.float32]
    global_rotation: npt.NDArray[np.float32]
    global_translation: npt.NDArray[np.float32]
    props: dict[str, Any]


@dataclasses.dataclass
class BodyModelsViserPoseMessage(_messages.Message, include_in_scene_serialization=True):
    name: str
    rest_vertices: npt.NDArray[np.float32] | None
    skinning_transforms: npt.NDArray[np.float32] | None
    pose_offsets: npt.NDArray[np.float32] | None
    global_rotation: npt.NDArray[np.float32]
    global_translation: npt.NDArray[np.float32]


@dataclasses.dataclass
class BodyModelsViserReadyMessage(_messages.Message, include_in_scene_serialization=False):
    pass


@dataclasses.dataclass
class _RuntimeState:
    ready_clients: set[int] = dataclasses.field(default_factory=set)
    installed_clients: set[int] = dataclasses.field(default_factory=set)
    models: dict[str, BodyModelsViserModelMessage] = dataclasses.field(default_factory=dict)
    poses: dict[str, BodyModelsViserPoseMessage] = dataclasses.field(default_factory=dict)


class BodyModelHandle:
    def __init__(
        self,
        scene: Any,
        name: str,
        model: Any,
        pose: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
        identity: dict[str, Any],
    ) -> None:
        self.scene = scene
        self.name = name
        self.model = model
        self.pose = pose
        self.identity = identity
        self.identity_keys = _parameter_keys(model.prepare_identity, pose.keys())
        self.pose_keys = _pose_keys(model, pose.keys())

    @property
    def global_rotation(self) -> Float[np.ndarray, "3"]:
        return self.pose["global_rotation"]

    @global_rotation.setter
    def global_rotation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_transform(global_rotation=value)

    @property
    def global_translation(self) -> Float[np.ndarray, "3"]:
        return self.pose["global_translation"]

    @global_translation.setter
    def global_translation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_transform(global_translation=value)

    def set_identity(self, **params: Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]) -> None:
        invalid = params.keys() - self.identity_keys
        if invalid:
            raise ValueError(f"Invalid identity parameter(s): {', '.join(sorted(invalid))}.")
        self._update_pose(params)
        self.identity = _prepare_identity(self.model, self.pose)
        prepared_pose = _prepare_pose(self.model, self.pose, self.identity)
        message = _pose_message(self.model, self.name, self.pose, self.identity, prepared_pose)
        _runtime_state(self.scene).poses[self.name] = message
        _queue_ready_clients(self.scene, message)

    def set_pose(self, **params: Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]) -> None:
        invalid = params.keys() - self.pose_keys
        if invalid:
            raise ValueError(f"Invalid pose parameter(s): {', '.join(sorted(invalid))}.")
        self._update_pose(params)
        prepared_pose = _prepare_pose(self.model, self.pose, self.identity)
        message = _pose_message(self.model, self.name, self.pose, self.identity, prepared_pose)
        _runtime_state(self.scene).poses[self.name] = message
        _queue_ready_clients(self.scene, message)

    def set_transform(self, **params: Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]) -> None:
        invalid = params.keys() - {"global_rotation", "global_translation"}
        if invalid:
            raise ValueError(f"Invalid transform parameter(s): {', '.join(sorted(invalid))}.")
        self._update_pose(params)
        message = BodyModelsViserPoseMessage(
            name=self.name,
            rest_vertices=None,
            skinning_transforms=None,
            pose_offsets=None,
            global_rotation=np.ascontiguousarray(self.pose["global_rotation"], dtype="<f4"),
            global_translation=np.ascontiguousarray(self.pose["global_translation"], dtype="<f4"),
        )
        _runtime_state(self.scene).poses[self.name] = message
        _queue_ready_clients(self.scene, message)

    def _update_pose(self, params: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]]) -> None:
        for key, value in params.items():
            self.pose[key] = np.asarray(value, dtype=np.float32).copy()

    def remove(self) -> None:
        state = _runtime_state(self.scene)
        del state.models[self.name]
        state.poses.pop(self.name, None)
        _queue_ready_clients(self.scene, _messages.RemoveSceneNodeMessage(self.name))


def _identity_property(key: str) -> property:
    def get(self: BodyModelHandle) -> np.ndarray:
        return self.pose[key]

    def set(self: BodyModelHandle, value: np.ndarray) -> None:
        self.set_identity(**{key: value})

    return property(get, set)


def _pose_property(key: str) -> property:
    def get(self: BodyModelHandle) -> np.ndarray:
        return self.pose[key]

    def set(self: BodyModelHandle, value: np.ndarray) -> None:
        self.set_pose(**{key: value})

    return property(get, set)


class AnnyBodyHandle(BodyModelHandle):
    shape = _identity_property("shape")
    body_pose = _pose_property("body_pose")
    head_pose = _pose_property("head_pose")
    hand_pose = _pose_property("hand_pose")


class FlameBodyHandle(BodyModelHandle):
    shape = _identity_property("shape")
    expression = _identity_property("expression")
    head_pose = _pose_property("head_pose")
    head_rotation = _pose_property("head_rotation")


class GarmentMeasurementsBodyHandle(BodyModelHandle):
    shape = _identity_property("shape")
    body_pose = _pose_property("body_pose")
    head_pose = _pose_property("head_pose")
    hand_pose = _pose_property("hand_pose")
    pelvis_rotation = _pose_property("pelvis_rotation")


class ManoBodyHandle(BodyModelHandle):
    shape = _identity_property("shape")
    hand_pose = _pose_property("hand_pose")
    wrist_rotation = _pose_property("wrist_rotation")


class MhrBodyHandle(BodyModelHandle):
    shape = _identity_property("shape")
    expression = _identity_property("expression")
    body_pose = _pose_property("body_pose")
    hand_pose = _pose_property("hand_pose")


class SkelBodyHandle(BodyModelHandle):
    shape = _identity_property("shape")
    body_pose = _pose_property("body_pose")


class SmplBodyHandle(BodyModelHandle):
    shape = _identity_property("shape")
    body_pose = _pose_property("body_pose")
    pelvis_rotation = _pose_property("pelvis_rotation")


class SmplhBodyHandle(BodyModelHandle):
    shape = _identity_property("shape")
    body_pose = _pose_property("body_pose")
    hand_pose = _pose_property("hand_pose")
    pelvis_rotation = _pose_property("pelvis_rotation")


class SmplxBodyHandle(BodyModelHandle):
    shape = _identity_property("shape")
    expression = _identity_property("expression")
    body_pose = _pose_property("body_pose")
    hand_pose = _pose_property("hand_pose")
    head_pose = _pose_property("head_pose")
    pelvis_rotation = _pose_property("pelvis_rotation")


class SomaBodyHandle(BodyModelHandle):
    shape = _identity_property("shape")
    body_pose = _pose_property("body_pose")
    head_pose = _pose_property("head_pose")
    hand_pose = _pose_property("hand_pose")


_HANDLE_TYPES = (
    (ANNY, AnnyBodyHandle),
    (FLAME, FlameBodyHandle),
    (GarmentMeasurements, GarmentMeasurementsBodyHandle),
    (MANO, ManoBodyHandle),
    (MHR, MhrBodyHandle),
    (SKEL, SkelBodyHandle),
    (SMPLH, SmplhBodyHandle),
    (SMPLX, SmplxBodyHandle),
    (SOMA, SomaBodyHandle),
    (SMPL, SmplBodyHandle),
)


def add_body_model(
    scene: Any,
    name: str,
    model: Any,
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
) -> BodyModelHandle:
    for model_type, handle_type in _HANDLE_TYPES:
        if isinstance(model, model_type):
            break
    else:
        raise TypeError(f"Unsupported body model {type(model).__name__}.")

    rest_pose = model.get_rest_pose()
    pose = {key: np.asarray(value, dtype=np.float32).copy() for key, value in rest_pose.items()}
    identity = _prepare_identity(model, pose)
    prepared_pose = _prepare_pose(model, pose, identity)
    skinning = model.prepare_skinning(identity=identity, pose=prepared_pose)
    rest_vertices = skinning["rest_vertices"]
    if "pose_offsets" in skinning:
        pose_offsets = skinning["pose_offsets"]
    else:
        pose_offsets = np.zeros_like(rest_vertices)
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
    message = BodyModelsViserModelMessage(
        name=name,
        vertex_count=int(rest_vertices.shape[0]),
        lbs_weights=np.ascontiguousarray(skinning["skin_weights"], dtype="<f4"),
        faces=np.ascontiguousarray(skinning["faces"], dtype="<u4"),
        rest_vertices=np.ascontiguousarray(rest_vertices, dtype="<f4"),
        skinning_transforms=np.ascontiguousarray(skinning["skinning_transforms"], dtype="<f4"),
        pose_offsets=np.ascontiguousarray(pose_offsets, dtype="<f4"),
        global_rotation=np.ascontiguousarray(pose["global_rotation"], dtype="<f4"),
        global_translation=np.ascontiguousarray(pose["global_translation"], dtype="<f4"),
        props=props,
    )
    state = _runtime_state(scene)
    state.models[name] = message
    state.poses.pop(name, None)
    _install_connected_clients(scene, state)
    _queue_ready_clients(scene, message)

    @scene._websock_interface.on_client_connect
    def _(client: Any) -> None:
        _install_client_runtime(client, state)

    return handle_type(scene, name, model, pose, identity)


def _runtime_state(scene: Any) -> _RuntimeState:
    websock = scene._websock_interface
    state = getattr(websock, "_body_models_viser", None)
    if state is not None:
        return state

    state = _RuntimeState()
    setattr(websock, "_body_models_viser", state)

    def ready(client_id: int, _: BodyModelsViserReadyMessage) -> None:
        state.ready_clients.add(client_id)
        client_state = websock._client_state_from_id[client_id]
        _replay_state(client_state, state)

    websock.register_handler(BodyModelsViserReadyMessage, ready)
    return state


def _install_connected_clients(scene: Any, state: _RuntimeState) -> None:
    for client_id, client_state in scene._websock_interface._client_state_from_id.items():
        if client_id not in state.installed_clients:
            state.installed_clients.add(client_id)
            install_message = _messages.RunJavascriptMessage(_install_javascript())
            client_state.message_buffer.push(install_message)
            client_state.message_buffer.flush()


def _install_client_runtime(client: Any, state: _RuntimeState) -> None:
    if client.client_id in state.installed_clients:
        return
    state.installed_clients.add(client.client_id)
    install_message = _messages.RunJavascriptMessage(_install_javascript())
    client.queue_message(install_message)
    client.get_message_buffer().flush()


def _queue_ready_clients(scene: Any, message: _messages.Message) -> None:
    state = _runtime_state(scene)
    for client_id in state.ready_clients:
        client_state = scene._websock_interface._client_state_from_id.get(client_id)
        if client_state is not None:
            client_state.message_buffer.push(message)


def _replay_state(client_state: Any, state: _RuntimeState) -> None:
    for name, message in state.models.items():
        client_state.message_buffer.push(message)
        if name in state.poses:
            client_state.message_buffer.push(state.poses[name])


def _install_javascript() -> str:
    ensure_client_is_built()
    source = (files(__package__) / "client" / "body-models-viser.js").read_text()
    wasm_path = files(__package__) / "client" / "body-models-viser.wasm"
    wasm = base64.b64encode(wasm_path.read_bytes()).decode("ascii")
    return f"""
(() => {{
  if (window.BodyModelsViser !== undefined) {{
    window.BodyModelsViser.ready();
    return;
  }}
  {source}
  window.BodyModelsViser = BodyModelsViser;
  window.BodyModelsViser.install({wasm!r});
}})();
"""


def _pose_message(
    model: Any,
    name: str,
    pose: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
    identity: dict[str, Any],
    prepared_pose: dict[str, Any],
) -> BodyModelsViserPoseMessage:
    skinning = model.prepare_skinning(identity=identity, pose=prepared_pose)
    skinning_rest_vertices = skinning["rest_vertices"]
    if "pose_offsets" in skinning:
        skinning_pose_offsets = skinning["pose_offsets"]
    else:
        skinning_pose_offsets = np.zeros_like(skinning_rest_vertices)

    return BodyModelsViserPoseMessage(
        name=name,
        rest_vertices=np.ascontiguousarray(skinning_rest_vertices, dtype="<f4"),
        skinning_transforms=np.ascontiguousarray(skinning["skinning_transforms"], dtype="<f4"),
        pose_offsets=np.ascontiguousarray(skinning_pose_offsets, dtype="<f4"),
        global_rotation=np.ascontiguousarray(pose["global_rotation"], dtype="<f4"),
        global_translation=np.ascontiguousarray(pose["global_translation"], dtype="<f4"),
    )


def _prepare_pose(
    model: Any,
    params: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
    identity: dict[str, Any],
) -> dict[str, Any]:
    if "pose" in inspect.signature(model.prepare_pose).parameters:
        pose = pack_pose(np, params["pelvis_rotation"], params["body_pose"], params["head_pose"], params["hand_pose"])
        return model.prepare_pose(pose, identity=identity)

    pose_params = {}
    for key in _parameter_keys(model.prepare_pose, params.keys()):
        pose_params[key] = np.zeros_like(params[key]) if key == "global_rotation" else params[key]
    return model.prepare_pose(**pose_params, identity=identity)


def _prepare_identity(
    model: Any,
    params: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
) -> dict[str, Any]:
    identity_params = {key: params[key] for key in _parameter_keys(model.prepare_identity, params.keys())}
    return model.prepare_identity(**identity_params)


def _parameter_keys(method: Callable[..., Any], keys: Iterable[str]) -> set[str]:
    parameters = inspect.signature(method).parameters
    return set(parameters) & set(keys)


def _pose_keys(model: Any, keys: Iterable[str]) -> set[str]:
    parameters = inspect.signature(model.prepare_pose).parameters
    if "pose" in parameters:
        return {"body_pose", "head_pose", "hand_pose", "pelvis_rotation"} & set(keys)
    return (set(parameters) & set(keys)) - {"global_rotation"}
