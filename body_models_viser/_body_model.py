from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

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

Params = dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]]


class BodyModelHandle:
    """Viser handle for one skinned body model.

    Subclasses declare which rest-pose keys are identity parameters and which
    are pose parameters; matching properties are generated automatically.
    """

    identity_keys: ClassVar[tuple[str, ...]] = ()
    pose_keys: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls) -> None:
        for key in cls.identity_keys:
            setattr(cls, key, _identity_property(key))
        for key in cls.pose_keys:
            setattr(cls, key, _pose_property(key))

    def __init__(self, scene: Any, name: str, model: Any, pose: Params) -> None:
        self.scene = scene
        self.name = name
        self.model = model
        self.pose = pose
        self.identity = self._prepare_identity()

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

    def set_identity(self, **params: np.ndarray) -> None:
        invalid = params.keys() - set(self.identity_keys)
        if invalid:
            raise ValueError(f"Invalid identity parameter(s): {', '.join(sorted(invalid))}.")
        self._update_pose(params)
        self.identity = self._prepare_identity()
        self._publish_pose()

    def set_pose(self, **params: np.ndarray) -> None:
        invalid = params.keys() - set(self.pose_keys)
        if invalid:
            raise ValueError(f"Invalid pose parameter(s): {', '.join(sorted(invalid))}.")
        self._update_pose(params)
        self._publish_pose()

    def set_transform(self, **params: np.ndarray) -> None:
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

    def _prepare_identity(self) -> Any:
        return self.model.prepare_identity(**{key: self.pose[key] for key in self.identity_keys})

    def _prepare_pose(self) -> Any:
        return self.model.prepare_pose(
            **{key: self.pose[key] for key in self.pose_keys}, identity=self.identity
        )

    def _publish_pose(self) -> None:
        skinning, pose_offsets = _skinning_arrays(self.model, self.identity, self._prepare_pose())
        message = BodyModelsViserPoseMessage(
            name=self.name,
            rest_vertices=np.ascontiguousarray(skinning["rest_vertices"], dtype="<f4"),
            skinning_transforms=np.ascontiguousarray(skinning["skinning_transforms"], dtype="<f4"),
            pose_offsets=np.ascontiguousarray(pose_offsets, dtype="<f4"),
            global_rotation=np.ascontiguousarray(self.pose["global_rotation"], dtype="<f4"),
            global_translation=np.ascontiguousarray(self.pose["global_translation"], dtype="<f4"),
        )
        state = _runtime.get_state(self.scene)
        state.poses[self.name] = message
        _runtime.broadcast(self.scene, message)

    def _update_pose(self, params: dict[str, np.ndarray]) -> None:
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
    identity_keys = ("shape",)
    pose_keys = ("body_pose", "head_pose", "hand_pose")

    def _prepare_pose(self) -> Any:
        # ANNY.prepare_pose requires global_rotation; the browser applies the
        # global transform in WASM, so it must be zeroed here.
        return self.model.prepare_pose(
            **{key: self.pose[key] for key in self.pose_keys},
            global_rotation=np.zeros_like(self.pose["global_rotation"]),
            identity=self.identity,
        )


class FlameBodyHandle(BodyModelHandle):
    identity_keys = ("shape", "expression")
    pose_keys = ("head_pose", "head_rotation")


class GarmentMeasurementsBodyHandle(BodyModelHandle):
    identity_keys = ("shape",)
    pose_keys = ("body_pose", "head_pose", "hand_pose", "pelvis_rotation")

    def _prepare_pose(self) -> Any:
        # GarmentMeasurements.prepare_pose takes a single packed pose array.
        pose = pack_pose(
            np,
            self.pose["pelvis_rotation"],
            self.pose["body_pose"],
            self.pose["head_pose"],
            self.pose["hand_pose"],
        )
        return self.model.prepare_pose(pose, identity=self.identity)


class ManoBodyHandle(BodyModelHandle):
    identity_keys = ("shape",)
    pose_keys = ("hand_pose", "wrist_rotation")


class MhrBodyHandle(BodyModelHandle):
    identity_keys = ("shape", "expression")
    pose_keys = ("body_pose", "head_pose", "hand_pose")


class SkelBodyHandle(BodyModelHandle):
    identity_keys = ("shape",)
    pose_keys = ("body_pose", "head_pose")


class SmplBodyHandle(BodyModelHandle):
    identity_keys = ("shape",)
    pose_keys = ("body_pose", "pelvis_rotation")


class SmplhBodyHandle(BodyModelHandle):
    identity_keys = ("shape",)
    pose_keys = ("body_pose", "hand_pose", "pelvis_rotation")


class SmplxBodyHandle(BodyModelHandle):
    identity_keys = ("shape", "expression")
    pose_keys = ("body_pose", "hand_pose", "head_pose", "pelvis_rotation")


class SomaBodyHandle(BodyModelHandle):
    identity_keys = ("shape",)
    pose_keys = ("body_pose", "head_pose", "hand_pose")

    def _prepare_pose(self) -> Any:
        # SOMA.prepare_pose requires global_rotation; the browser applies the
        # global transform in WASM, so it must be zeroed here.
        return self.model.prepare_pose(
            **{key: self.pose[key] for key in self.pose_keys},
            global_rotation=np.zeros_like(self.pose["global_rotation"]),
            identity=self.identity,
        )


_HANDLE_TYPES: dict[type, type[BodyModelHandle]] = {
    ANNY: AnnyBodyHandle,
    FLAME: FlameBodyHandle,
    GarmentMeasurements: GarmentMeasurementsBodyHandle,
    MANO: ManoBodyHandle,
    MHR: MhrBodyHandle,
    SKEL: SkelBodyHandle,
    SMPL: SmplBodyHandle,
    SMPLH: SmplhBodyHandle,
    SMPLX: SmplxBodyHandle,
    SOMA: SomaBodyHandle,
}


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
    handle_type = _HANDLE_TYPES.get(type(model))
    if handle_type is None:
        raise TypeError(f"Unsupported body model {type(model).__name__}.")

    rest_pose = model.get_rest_pose()
    pose = {key: np.asarray(value, dtype=np.float32).copy() for key, value in rest_pose.items()}
    handle = handle_type(scene, name, model, pose)
    skinning, pose_offsets = _skinning_arrays(model, handle.identity, handle._prepare_pose())
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
    return handle


def _skinning_arrays(model: Any, identity: Any, prepared_pose: Any) -> tuple[Any, np.ndarray]:
    skinning = model.prepare_skinning(identity=identity, pose=prepared_pose)
    pose_offsets = skinning.get("pose_offsets")
    if pose_offsets is None:
        pose_offsets = np.zeros_like(skinning["rest_vertices"])
    return skinning, pose_offsets
