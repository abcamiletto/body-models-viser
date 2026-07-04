from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
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

from . import _runtime
from ._runtime import BodyModelsViserModelMessage, BodyModelsViserPoseMessage


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
        self._publish_pose()

    def set_pose(self, **params: Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]) -> None:
        invalid = params.keys() - self.pose_keys
        if invalid:
            raise ValueError(f"Invalid pose parameter(s): {', '.join(sorted(invalid))}.")
        self._update_pose(params)
        self._publish_pose()

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
        state = _runtime.get_state(self.scene)
        # Send the slim message to live clients, but keep full skinning data in
        # the stored message: it is the single source of truth replayed to late
        # clients and new offline exports.
        stored = state.poses.get(self.name)
        state.poses[self.name] = message if stored is None else dataclasses.replace(
            stored,
            global_rotation=message.global_rotation,
            global_translation=message.global_translation,
        )
        _runtime.broadcast(self.scene, message)

    def remove(self) -> None:
        state = _runtime.get_state(self.scene)
        del state.models[self.name]
        state.poses.pop(self.name, None)
        _runtime.broadcast(self.scene, _messages.RemoveSceneNodeMessage(self.name))

    def _publish_pose(self) -> None:
        prepared_pose = _prepare_pose(self.model, self.pose, self.identity)
        message = _pose_message(self.model, self.name, self.pose, self.identity, prepared_pose)
        state = _runtime.get_state(self.scene)
        state.poses[self.name] = message
        _runtime.broadcast(self.scene, message)

    def _update_pose(self, params: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]]) -> None:
        for key, value in params.items():
            self.pose[key] = np.asarray(value, dtype=np.float32).copy()


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
    skinning, pose_offsets = _skinning_arrays(model, identity, prepared_pose)
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
        vertex_count=int(skinning["rest_vertices"].shape[0]),
        lbs_weights=np.ascontiguousarray(skinning["skin_weights"], dtype="<f4"),
        faces=np.ascontiguousarray(skinning["faces"], dtype="<u4"),
        rest_vertices=np.ascontiguousarray(skinning["rest_vertices"], dtype="<f4"),
        skinning_transforms=np.ascontiguousarray(skinning["skinning_transforms"], dtype="<f4"),
        pose_offsets=np.ascontiguousarray(pose_offsets, dtype="<f4"),
        global_rotation=np.ascontiguousarray(pose["global_rotation"], dtype="<f4"),
        global_translation=np.ascontiguousarray(pose["global_translation"], dtype="<f4"),
        props=props,
    )
    state = _runtime.get_state(scene)
    state.models[name] = message
    state.poses.pop(name, None)
    _runtime.broadcast(scene, message)
    return handle_type(scene, name, model, pose, identity)


def _pose_message(
    model: Any,
    name: str,
    pose: dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]],
    identity: dict[str, Any],
    prepared_pose: dict[str, Any],
) -> BodyModelsViserPoseMessage:
    skinning, pose_offsets = _skinning_arrays(model, identity, prepared_pose)
    return BodyModelsViserPoseMessage(
        name=name,
        rest_vertices=np.ascontiguousarray(skinning["rest_vertices"], dtype="<f4"),
        skinning_transforms=np.ascontiguousarray(skinning["skinning_transforms"], dtype="<f4"),
        pose_offsets=np.ascontiguousarray(pose_offsets, dtype="<f4"),
        global_rotation=np.ascontiguousarray(pose["global_rotation"], dtype="<f4"),
        global_translation=np.ascontiguousarray(pose["global_translation"], dtype="<f4"),
    )


def _skinning_arrays(model: Any, identity: dict[str, Any], prepared_pose: dict[str, Any]) -> tuple[Any, np.ndarray]:
    skinning = model.prepare_skinning(identity=identity, pose=prepared_pose)
    pose_offsets = skinning.get("pose_offsets")
    if pose_offsets is None:
        pose_offsets = np.zeros_like(skinning["rest_vertices"])
    return skinning, pose_offsets


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
