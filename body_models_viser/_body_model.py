from __future__ import annotations

import dataclasses
from typing import Any

import body_models
import numpy as np
from jaxtyping import Float
from viser import _messages

from . import _runtime
from ._runtime import BodyModelsViserModelMessage, BodyModelsViserPoseMessage

Params = dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]]


class BodyModelHandle:
    """Viser handle for one skinned body model."""

    def __init__(self, scene: Any, name: str, model: body_models.SkinnedModel, params: Params) -> None:
        self.scene = scene
        self.name = name
        self.model = model
        self.params = params
        self._prepared_identity = self._prepare_identity()

    def __getattr__(self, key: str) -> np.ndarray:
        params = self.__dict__.get("params")
        if params is not None and key in params:
            return params[key]
        raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        model = self.__dict__.get("model")
        if model is not None:
            if key in model.identity_keys:
                self.set_identity(**{key: value})
                return
            if key in model.pose_keys:
                self.set_pose(**{key: value})
                return
            if key in model.transform_keys:
                self.set_transform(**{key: value})
                return
        super().__setattr__(key, value)

    def set_identity(self, **params: np.ndarray) -> None:
        invalid = params.keys() - set(self.model.identity_keys)
        if invalid:
            raise ValueError(f"Invalid identity parameter(s): {', '.join(sorted(invalid))}.")
        self._update_params(params)
        self._prepared_identity = self._prepare_identity()
        self._publish_pose()

    def set_pose(self, **params: np.ndarray) -> None:
        invalid = params.keys() - set(self.model.pose_keys)
        if invalid:
            raise ValueError(f"Invalid pose parameter(s): {', '.join(sorted(invalid))}.")
        self._update_params(params)
        self._publish_pose()

    def set_transform(self, **params: np.ndarray) -> None:
        invalid = params.keys() - set(self.model.transform_keys)
        if invalid:
            raise ValueError(f"Invalid transform parameter(s): {', '.join(sorted(invalid))}.")
        self._update_params(params)
        message = BodyModelsViserPoseMessage(
            name=self.name,
            rest_vertices=None,
            skinning_transforms=None,
            pose_offsets=None,
            global_rotation=np.ascontiguousarray(self.params["global_rotation"], dtype="<f4"),
            global_translation=np.ascontiguousarray(self.params["global_translation"], dtype="<f4"),
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
        identity_params = {key: self.params[key] for key in self.model.identity_keys}
        return self.model.prepare_identity(**identity_params)

    def _prepare_pose(self) -> Any:
        pose_params = {key: self.params[key] for key in self.model.pose_keys}
        return self.model.prepare_pose(**pose_params, identity=self._prepared_identity)

    def _publish_pose(self) -> None:
        skinning, pose_offsets = _skinning_arrays(self.model, self._prepared_identity, self._prepare_pose())
        message = BodyModelsViserPoseMessage(
            name=self.name,
            rest_vertices=np.ascontiguousarray(skinning["rest_vertices"], dtype="<f4"),
            skinning_transforms=np.ascontiguousarray(skinning["skinning_transforms"], dtype="<f4"),
            pose_offsets=np.ascontiguousarray(pose_offsets, dtype="<f4"),
            global_rotation=np.ascontiguousarray(self.params["global_rotation"], dtype="<f4"),
            global_translation=np.ascontiguousarray(self.params["global_translation"], dtype="<f4"),
        )
        state = _runtime.get_state(self.scene)
        state.poses[self.name] = message
        _runtime.broadcast(self.scene, message)

    def _update_params(self, params: dict[str, np.ndarray]) -> None:
        for key, value in params.items():
            self.params[key] = np.asarray(value, dtype=np.float32).copy()


def add_body_model(
    scene: Any,
    name: str,
    model: body_models.SkinnedModel,
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
    if not isinstance(model, body_models.SkinnedModel):
        raise TypeError(f"Expected body_models.SkinnedModel, got {type(model).__name__}.")

    rest_pose = model.get_rest_pose()
    params = {key: np.asarray(value, dtype=np.float32).copy() for key, value in rest_pose.items()}
    handle = BodyModelHandle(scene, name, model, params)
    skinning, pose_offsets = _skinning_arrays(model, handle._prepared_identity, handle._prepare_pose())
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
        skin_weights=np.ascontiguousarray(skinning["skin_weights"], dtype="<f4"),
        faces=np.ascontiguousarray(skinning["faces"], dtype="<u4"),
        rest_vertices=np.ascontiguousarray(skinning["rest_vertices"], dtype="<f4"),
        skinning_transforms=np.ascontiguousarray(skinning["skinning_transforms"], dtype="<f4"),
        pose_offsets=np.ascontiguousarray(pose_offsets, dtype="<f4"),
        global_rotation=np.ascontiguousarray(params["global_rotation"], dtype="<f4"),
        global_translation=np.ascontiguousarray(params["global_translation"], dtype="<f4"),
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
