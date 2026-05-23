from __future__ import annotations

import base64
import dataclasses
import hashlib
from importlib.resources import files
from typing import Any

import numpy as np
import numpy.typing as npt
from body_models.anny import pose as anny_pose
from body_models.anny.numpy import ANNY
from body_models.flame.numpy import FLAME
from body_models.mano.numpy import MANO
from body_models.mhr import pose as mhr_pose
from body_models.mhr.backends import core as mhr_core
from body_models.mhr.numpy import MHR
from body_models.skel.numpy import SKEL
from body_models.smpl.numpy import SMPL
from body_models.smplh.numpy import SMPLH
from body_models.smplx.numpy import SMPLX
from body_models.soma import pose as soma_pose
from body_models.soma.backends import core as soma_core
from body_models.soma.numpy import SOMA
from jaxtyping import Float
from viser import _messages

from ._client_autobuild import ensure_client_is_built

RUNTIME_VERSION = "0.0.2"


@dataclasses.dataclass
class BodyModelsViserModelMessage(_messages.Message, include_in_scene_serialization=True):
    model_type: str
    name: str
    vertex_count: int
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


@dataclasses.dataclass
class BodyModelsViserReadyMessage(_messages.Message, include_in_scene_serialization=False):
    pass


@dataclasses.dataclass
class _RuntimeState:
    ready_clients: set[int] = dataclasses.field(default_factory=set)
    installed_clients: set[int] = dataclasses.field(default_factory=set)
    models: dict[str, BodyModelsViserModelMessage] = dataclasses.field(default_factory=dict)
    poses: dict[str, BodyModelsViserPoseMessage] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class _RuntimeInputs:
    model_type: str
    lbs_weights: np.ndarray
    rest_joints: np.ndarray
    rest_vertices: np.ndarray
    joint_transforms: np.ndarray
    pose_offsets: np.ndarray


class BodyModelHandle:
    def __init__(
        self,
        scene: Any,
        name: str,
        model: Any,
        pose: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
    ) -> None:
        self.scene = scene
        self.name = name
        self.model = model
        self.pose = pose

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
        identity_changed = bool({"shape", "expression", "scale_params"} & params.keys())
        pose_changed = identity_changed or bool(_pose_keys(self.model) & params.keys())
        for key, value in params.items():
            self.pose[key] = np.asarray(value, dtype=np.float32).copy()
        identity = _prepare_identity(self.model, self.pose) if identity_changed else None
        prepared_pose = _prepare_pose(self.model, self.pose, identity) if pose_changed else None
        message = _pose_message(self.model, self.name, self.pose, identity, prepared_pose)
        _runtime_state(self.scene).poses[self.name] = message
        _queue_ready_clients(self.scene, message)

    def remove(self) -> None:
        state = _runtime_state(self.scene)
        state.models.pop(self.name, None)
        state.poses.pop(self.name, None)
        _queue_ready_clients(self.scene, _messages.RemoveSceneNodeMessage(self.name))


class AnnyBodyHandle(BodyModelHandle):
    @property
    def shape(self) -> Float[np.ndarray, "6"]:
        return self.pose["shape"]

    @shape.setter
    def shape(self, value: Float[np.ndarray, "6"]) -> None:
        self.set_pose(shape=value)

    @property
    def body_pose(self) -> Float[np.ndarray, "64 3"]:
        return self.pose["body_pose"]

    @body_pose.setter
    def body_pose(self, value: Float[np.ndarray, "64 3"]) -> None:
        self.set_pose(body_pose=value)

    @property
    def head_pose(self) -> Float[np.ndarray, "60 3"]:
        return self.pose["head_pose"]

    @head_pose.setter
    def head_pose(self, value: Float[np.ndarray, "60 3"]) -> None:
        self.set_pose(head_pose=value)

    @property
    def hand_pose(self) -> Float[np.ndarray, "38 3"]:
        return self.pose["hand_pose"]

    @hand_pose.setter
    def hand_pose(self, value: Float[np.ndarray, "38 3"]) -> None:
        self.set_pose(hand_pose=value)


class FlameBodyHandle(BodyModelHandle):
    @property
    def shape(self) -> Float[np.ndarray, "300"]:
        return self.pose["shape"]

    @shape.setter
    def shape(self, value: Float[np.ndarray, "300"]) -> None:
        self.set_pose(shape=value)

    @property
    def expression(self) -> Float[np.ndarray, "100"]:
        return self.pose["expression"]

    @expression.setter
    def expression(self, value: Float[np.ndarray, "100"]) -> None:
        self.set_pose(expression=value)

    @property
    def head_pose(self) -> Float[np.ndarray, "4 3"]:
        return self.pose["head_pose"]

    @head_pose.setter
    def head_pose(self, value: Float[np.ndarray, "4 3"]) -> None:
        self.set_pose(head_pose=value)

    @property
    def head_rotation(self) -> Float[np.ndarray, "3"]:
        return self.pose["head_rotation"]

    @head_rotation.setter
    def head_rotation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_pose(head_rotation=value)


class ManoBodyHandle(BodyModelHandle):
    @property
    def shape(self) -> Float[np.ndarray, "10"]:
        return self.pose["shape"]

    @shape.setter
    def shape(self, value: Float[np.ndarray, "10"]) -> None:
        self.set_pose(shape=value)

    @property
    def hand_pose(self) -> Float[np.ndarray, "15 3"]:
        return self.pose["hand_pose"]

    @hand_pose.setter
    def hand_pose(self, value: Float[np.ndarray, "15 3"]) -> None:
        self.set_pose(hand_pose=value)

    @property
    def wrist_rotation(self) -> Float[np.ndarray, "3"]:
        return self.pose["wrist_rotation"]

    @wrist_rotation.setter
    def wrist_rotation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_pose(wrist_rotation=value)


class MhrBodyHandle(BodyModelHandle):
    @property
    def shape(self) -> Float[np.ndarray, "45"]:
        return self.pose["shape"]

    @shape.setter
    def shape(self, value: Float[np.ndarray, "45"]) -> None:
        self.set_pose(shape=value)

    @property
    def expression(self) -> Float[np.ndarray, "72"]:
        return self.pose["expression"]

    @expression.setter
    def expression(self, value: Float[np.ndarray, "72"]) -> None:
        self.set_pose(expression=value)

    @property
    def body_pose(self) -> Float[np.ndarray, "100"]:
        return self.pose["body_pose"]

    @body_pose.setter
    def body_pose(self, value: Float[np.ndarray, "100"]) -> None:
        self.set_pose(body_pose=value)

    @property
    def hand_pose(self) -> Float[np.ndarray, "104"]:
        return self.pose["hand_pose"]

    @hand_pose.setter
    def hand_pose(self, value: Float[np.ndarray, "104"]) -> None:
        self.set_pose(hand_pose=value)


class SkelBodyHandle(BodyModelHandle):
    @property
    def shape(self) -> Float[np.ndarray, "10"]:
        return self.pose["shape"]

    @shape.setter
    def shape(self, value: Float[np.ndarray, "10"]) -> None:
        self.set_pose(shape=value)

    @property
    def body_pose(self) -> Float[np.ndarray, "46"]:
        return self.pose["body_pose"]

    @body_pose.setter
    def body_pose(self, value: Float[np.ndarray, "46"]) -> None:
        self.set_pose(body_pose=value)


class SmplBodyHandle(BodyModelHandle):
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


class SmplhBodyHandle(BodyModelHandle):
    @property
    def shape(self) -> Float[np.ndarray, "10"]:
        return self.pose["shape"]

    @shape.setter
    def shape(self, value: Float[np.ndarray, "10"]) -> None:
        self.set_pose(shape=value)

    @property
    def body_pose(self) -> Float[np.ndarray, "21 3"]:
        return self.pose["body_pose"]

    @body_pose.setter
    def body_pose(self, value: Float[np.ndarray, "21 3"]) -> None:
        self.set_pose(body_pose=value)

    @property
    def hand_pose(self) -> Float[np.ndarray, "30 3"]:
        return self.pose["hand_pose"]

    @hand_pose.setter
    def hand_pose(self, value: Float[np.ndarray, "30 3"]) -> None:
        self.set_pose(hand_pose=value)

    @property
    def pelvis_rotation(self) -> Float[np.ndarray, "3"]:
        return self.pose["pelvis_rotation"]

    @pelvis_rotation.setter
    def pelvis_rotation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_pose(pelvis_rotation=value)


class SmplxBodyHandle(BodyModelHandle):
    @property
    def shape(self) -> Float[np.ndarray, "10"]:
        return self.pose["shape"]

    @shape.setter
    def shape(self, value: Float[np.ndarray, "10"]) -> None:
        self.set_pose(shape=value)

    @property
    def expression(self) -> Float[np.ndarray, "10"]:
        return self.pose["expression"]

    @expression.setter
    def expression(self, value: Float[np.ndarray, "10"]) -> None:
        self.set_pose(expression=value)

    @property
    def body_pose(self) -> Float[np.ndarray, "21 3"]:
        return self.pose["body_pose"]

    @body_pose.setter
    def body_pose(self, value: Float[np.ndarray, "21 3"]) -> None:
        self.set_pose(body_pose=value)

    @property
    def hand_pose(self) -> Float[np.ndarray, "30 3"]:
        return self.pose["hand_pose"]

    @hand_pose.setter
    def hand_pose(self, value: Float[np.ndarray, "30 3"]) -> None:
        self.set_pose(hand_pose=value)

    @property
    def head_pose(self) -> Float[np.ndarray, "3 3"]:
        return self.pose["head_pose"]

    @head_pose.setter
    def head_pose(self, value: Float[np.ndarray, "3 3"]) -> None:
        self.set_pose(head_pose=value)

    @property
    def pelvis_rotation(self) -> Float[np.ndarray, "3"]:
        return self.pose["pelvis_rotation"]

    @pelvis_rotation.setter
    def pelvis_rotation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_pose(pelvis_rotation=value)


class SomaBodyHandle(BodyModelHandle):
    @property
    def shape(self) -> Float[np.ndarray, "128"]:
        return self.pose["shape"]

    @shape.setter
    def shape(self, value: Float[np.ndarray, "128"]) -> None:
        self.set_pose(shape=value)

    @property
    def body_pose(self) -> Float[np.ndarray, "23 3"]:
        return self.pose["body_pose"]

    @body_pose.setter
    def body_pose(self, value: Float[np.ndarray, "23 3"]) -> None:
        self.set_pose(body_pose=value)

    @property
    def head_pose(self) -> Float[np.ndarray, "5 3"]:
        return self.pose["head_pose"]

    @head_pose.setter
    def head_pose(self, value: Float[np.ndarray, "5 3"]) -> None:
        self.set_pose(head_pose=value)

    @property
    def hand_pose(self) -> Float[np.ndarray, "48 3"]:
        return self.pose["hand_pose"]

    @hand_pose.setter
    def hand_pose(self, value: Float[np.ndarray, "48 3"]) -> None:
        self.set_pose(hand_pose=value)


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
    if not isinstance(model, (ANNY, FLAME, MANO, MHR, SKEL, SMPL, SMPLH, SMPLX, SOMA)):
        raise TypeError(f"Unsupported body model {type(model).__name__}.")
    if getattr(model, "rotation_type", "axis_angle") != "axis_angle":
        raise ValueError("body-models-viser supports axis-angle parameters only.")

    pose = {key: np.asarray(value, dtype=np.float32).copy() for key, value in model.get_rest_pose().items()}
    identity = _prepare_identity(model, pose)
    prepared_pose = _prepare_pose(model, pose, identity)
    runtime_inputs = _runtime_inputs(model, identity, prepared_pose)
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
        model_type=runtime_inputs.model_type,
        name=name,
        vertex_count=int(identity["rest_vertices"].shape[0]),
        lbs_weights=_array(runtime_inputs.lbs_weights, np.float32),
        faces=_array(model.faces, np.uint32),
        rest_joints=_array(runtime_inputs.rest_joints, np.float32),
        rest_vertices=_array(runtime_inputs.rest_vertices, np.float32),
        joint_transforms=_array(runtime_inputs.joint_transforms, np.float32),
        pose_offsets=_array(runtime_inputs.pose_offsets, np.float32),
        global_rotation=_array(_global_rotation(model, pose), np.float32),
        global_translation=_array(pose["global_translation"], np.float32),
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

    return _make_handle(scene, name, model, pose)


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


def _make_handle(
    scene: Any,
    name: str,
    model: Any,
    pose: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
) -> BodyModelHandle:
    if isinstance(model, ANNY):
        return AnnyBodyHandle(scene, name, model, pose)
    if isinstance(model, FLAME):
        return FlameBodyHandle(scene, name, model, pose)
    if isinstance(model, MANO):
        return ManoBodyHandle(scene, name, model, pose)
    if isinstance(model, MHR):
        return MhrBodyHandle(scene, name, model, pose)
    if isinstance(model, SKEL):
        return SkelBodyHandle(scene, name, model, pose)
    if isinstance(model, SMPLH):
        return SmplhBodyHandle(scene, name, model, pose)
    if isinstance(model, SMPLX):
        return SmplxBodyHandle(scene, name, model, pose)
    if isinstance(model, SOMA):
        return SomaBodyHandle(scene, name, model, pose)
    return SmplBodyHandle(scene, name, model, pose)


def _install_connected_clients(scene: Any, state: _RuntimeState) -> None:
    for client_id, client_state in scene._websock_interface._client_state_from_id.items():
        if client_id not in state.installed_clients:
            state.installed_clients.add(client_id)
            client_state.message_buffer.push(_messages.RunJavascriptMessage(_install_javascript()))
            client_state.message_buffer.flush()


def _install_client_runtime(client: Any, state: _RuntimeState) -> None:
    if client.client_id in state.installed_clients:
        return
    state.installed_clients.add(client.client_id)
    client.queue_message(_messages.RunJavascriptMessage(_install_javascript()))
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
    wasm_bytes = _wasm_bytes()
    wasm = base64.b64encode(wasm_bytes).decode("ascii")
    build_id = hashlib.sha256(source.encode() + wasm_bytes).hexdigest()
    return f"""
(() => {{
  const version = {RUNTIME_VERSION!r};
  const buildId = {build_id!r};
  if (window.BodyModelsViser !== undefined) {{
    if (window.BodyModelsViser.version !== version || window.BodyModelsViser.buildId !== buildId) {{
      throw new Error(`body-models-viser runtime mismatch: installed ${{window.BodyModelsViser.version}}/${{window.BodyModelsViser.buildId}}, requested ${{version}}/${{buildId}}`);
    }}
    window.BodyModelsViser.ready();
    return;
  }}
  {source}
  window.BodyModelsViser = BodyModelsViser;
  window.BodyModelsViser.install({wasm!r}, version, buildId);
}})();
"""


def _wasm_bytes() -> bytes:
    ensure_client_is_built()
    return (files(__package__) / "client" / "body-models-viser.wasm").read_bytes()


def _pose_message(
    model: Any,
    name: str,
    pose: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
    identity: dict[str, Any] | None,
    prepared_pose: dict[str, Any] | None,
) -> BodyModelsViserPoseMessage:
    runtime_inputs = None
    if identity is not None and prepared_pose is not None:
        runtime_inputs = _runtime_inputs(model, identity, prepared_pose)
    return BodyModelsViserPoseMessage(
        name=name,
        rest_joints=None if runtime_inputs is None else _array(runtime_inputs.rest_joints, np.float32),
        rest_vertices=None if runtime_inputs is None else _array(runtime_inputs.rest_vertices, np.float32),
        joint_transforms=None if runtime_inputs is None else _array(runtime_inputs.joint_transforms, np.float32),
        pose_offsets=None if runtime_inputs is None else _array(runtime_inputs.pose_offsets, np.float32),
        global_rotation=_array(_global_rotation(model, pose), np.float32),
        global_translation=_array(pose["global_translation"], np.float32),
    )


def _prepare_pose(
    model: Any,
    params: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
    identity: dict[str, Any] | None,
) -> dict[str, Any]:
    identity = _prepare_identity(model, params) if identity is None else identity
    if isinstance(model, SMPL):
        return model.prepare_pose(params["body_pose"], params["pelvis_rotation"], identity=identity)
    if isinstance(model, SMPLH):
        return model.prepare_pose(params["body_pose"], params["hand_pose"], params["pelvis_rotation"], identity=identity)
    if isinstance(model, SMPLX):
        return model.prepare_pose(
            params["body_pose"],
            params["hand_pose"],
            params["head_pose"],
            params["pelvis_rotation"],
            identity=identity,
        )
    if isinstance(model, MANO):
        return model.prepare_pose(params["hand_pose"], params["wrist_rotation"], identity=identity)
    if isinstance(model, FLAME):
        return model.prepare_pose(params["head_pose"], params["head_rotation"], identity=identity)
    if isinstance(model, SKEL):
        return model.prepare_pose(params["body_pose"], identity=identity)
    if isinstance(model, ANNY):
        pose = anny_pose.pack_pose(
            np,
            params["global_rotation"],
            params["body_pose"],
            params["head_pose"],
            params["hand_pose"],
        )
        return model.prepare_pose(pose, identity=identity)
    if isinstance(model, MHR):
        return model.prepare_pose(mhr_pose.pack_pose(np, params["body_pose"], params["hand_pose"]))
    if isinstance(model, SOMA):
        pose = soma_pose.pack_pose(
            np,
            params["global_rotation"],
            params["body_pose"],
            params["head_pose"],
            params["hand_pose"],
        )
        return model.prepare_pose(pose)
    raise TypeError(f"Unsupported body model {type(model).__name__}.")


def _prepare_identity(
    model: Any,
    params: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
) -> dict[str, Any]:
    if isinstance(model, (FLAME, MHR, SMPLX)):
        return model.prepare_identity(params["shape"], expression=params["expression"])
    if isinstance(model, SOMA):
        return model.prepare_identity(params["shape"], scale_params=params.get("scale_params"))
    return model.prepare_identity(params["shape"])


def _runtime_inputs(
    model: Any,
    identity: dict[str, Any],
    prepared_pose: dict[str, Any],
) -> _RuntimeInputs:
    if isinstance(model, ANNY):
        return _RuntimeInputs(
            model_type="skin",
            lbs_weights=model.weights.lbs_weights,
            rest_joints=np.zeros((prepared_pose["bone_transforms"].shape[0], 3), dtype=np.float32),
            rest_vertices=identity["rest_vertices"],
            joint_transforms=prepared_pose["bone_transforms"],
            pose_offsets=np.zeros_like(identity["rest_vertices"]),
        )
    if isinstance(model, MHR):
        rest_vertices = identity["rest_vertices"] * 0.01
        pose_offsets = mhr_core.apply_pose_correctives(
            prepared_pose["joint_params"],
            model.weights.corrective_W1,
            model.weights.corrective_W2,
            xp=np,
        ) * 0.01
        joint_linear = prepared_pose["joint_rotations"] * prepared_pose["joint_scales"][..., None]
        linear = np.einsum("jik,jkl->jil", joint_linear, model.weights.bind_inv_linear)
        translation = (
            np.einsum("jik,jk->ji", joint_linear, model.weights.bind_inv_translation)
            + prepared_pose["joint_translations"]
        ) * 0.01
        return _RuntimeInputs(
            model_type="skin",
            lbs_weights=model.skin_weights,
            rest_joints=np.zeros((linear.shape[0], 3), dtype=np.float32),
            rest_vertices=rest_vertices,
            joint_transforms=_transforms(linear, translation),
            pose_offsets=pose_offsets,
        )
    if isinstance(model, SOMA):
        pose_offsets = soma_core.apply_pose_correctives(model.weights, prepared_pose["pose_rot_full"], xp=np)
        if model.weights.vertex_map is not None:
            pose_offsets = pose_offsets[model.weights.vertex_map]
        world = soma_core._pose_skeleton_from_oriented_pose(
            xp=np,
            world_bind_pose=identity["world_bind_pose"],
            kinematic_fronts=model.weights.topology.kinematic_fronts_full,
            parents_full=model.weights.topology.parents_full,
            pose_rot_full=prepared_pose["pose_rot_full"],
        )
        transforms = world @ identity["inverse_world_bind_pose"]
        transforms[:, :3, 3] *= 0.01
        return _RuntimeInputs(
            model_type="skin",
            lbs_weights=model.weights.skin_weights_active,
            rest_joints=np.zeros((transforms.shape[0], 3), dtype=np.float32),
            rest_vertices=identity["bind_shape_active"] * 0.01,
            joint_transforms=transforms,
            pose_offsets=pose_offsets * 0.01,
        )
    if isinstance(model, SKEL):
        return _RuntimeInputs(
            model_type="smpl",
            lbs_weights=model.weights.skin_weights,
            rest_joints=identity["rest_joints"],
            rest_vertices=identity["rest_vertices"],
            joint_transforms=prepared_pose["joint_transforms"],
            pose_offsets=prepared_pose["pose_offsets"],
        )
    return _RuntimeInputs(
        model_type="smpl",
        lbs_weights=model.weights.lbs_weights,
        rest_joints=identity["rest_joints"],
        rest_vertices=identity["rest_vertices"],
        joint_transforms=prepared_pose["joint_transforms"],
        pose_offsets=prepared_pose["pose_offsets"],
    )


def _pose_keys(model: Any) -> set[str]:
    if isinstance(model, (ANNY, SOMA)):
        return {"body_pose", "head_pose", "hand_pose", "global_rotation"}
    if isinstance(model, MHR):
        return {"body_pose", "hand_pose"}
    if isinstance(model, (SMPLH, SMPLX)):
        return {"body_pose", "hand_pose", "head_pose", "pelvis_rotation"}
    if isinstance(model, MANO):
        return {"hand_pose", "wrist_rotation"}
    if isinstance(model, FLAME):
        return {"head_pose", "head_rotation"}
    return {"body_pose", "pelvis_rotation"}


def _transforms(linear: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transforms = np.zeros((*linear.shape[:-2], 4, 4), dtype=np.float32)
    transforms[..., :3, :3] = linear
    transforms[..., :3, 3] = translation
    transforms[..., 3, 3] = 1.0
    return transforms


def _global_rotation(
    model: Any,
    pose: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
) -> np.ndarray:
    if isinstance(model, (ANNY, SOMA)):
        return np.zeros(3, dtype=np.float32)
    return pose["global_rotation"]


def _array(array: Any, dtype: Any) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<"))
