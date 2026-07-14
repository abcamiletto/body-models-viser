from __future__ import annotations

import dataclasses
from typing import Any

import body_models
import numpy as np
from body_models.mhr.numpy import MHR
from body_models.skel.numpy import SKEL
from body_models.soma.numpy import SOMA
from jaxtyping import Float
from viser import _messages

from . import _runtime
from ._runtime import (
    BodyModelsViserAssetMessage,
    BodyModelsViserIdentityMessage,
    BodyModelsViserModelMessage,
    BodyModelsViserPoseMessage,
    BodyModelsViserTransformMessage,
)

Params = dict[str, Float[np.ndarray, "dim"] | Float[np.ndarray, "joints 3"]]
_SERVER_ONLY_CORRECTIVE_MODELS = (MHR, SOMA)
_NONSTANDARD_CORRECTIVE_MODELS = (SKEL,)


class BodyModelHandle:
    """Viser handle for one skinned body model."""

    def __init__(
        self,
        scene: Any,
        name: str,
        model: body_models.SkinnedModel,
        params: Params,
        *,
        use_pose_correctives: bool,
    ) -> None:
        self.scene = scene
        self.name = name
        self.model = model
        self.params = params
        self.use_pose_correctives = use_pose_correctives
        self._asset_key = (model, use_pose_correctives)
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
        skinning, coefficients = self._prepare_skinning()
        message = BodyModelsViserIdentityMessage(
            name=self.name,
            rest_vertices=_f32(skinning["rest_vertices"]),
            skinning_transforms=_f32(skinning["skinning_transforms"]),
            pose_coefficients=coefficients,
        )
        state = _runtime.get_state(self.scene)
        state.models[self.name] = dataclasses.replace(
            state.models[self.name],
            rest_vertices=message.rest_vertices,
            skinning_transforms=message.skinning_transforms,
            pose_coefficients=message.pose_coefficients,
        )
        _runtime.broadcast(self.scene, message)

    def set_pose(self, **params: np.ndarray) -> None:
        invalid = params.keys() - set(self.model.pose_keys)
        if invalid:
            raise ValueError(f"Invalid pose parameter(s): {', '.join(sorted(invalid))}.")
        self._update_params(params)
        skinning, coefficients = self._prepare_skinning()
        message = BodyModelsViserPoseMessage(
            name=self.name,
            skinning_transforms=_f32(skinning["skinning_transforms"]),
            pose_coefficients=coefficients,
        )
        state = _runtime.get_state(self.scene)
        state.models[self.name] = dataclasses.replace(
            state.models[self.name],
            skinning_transforms=message.skinning_transforms,
            pose_coefficients=message.pose_coefficients,
        )
        _runtime.broadcast(self.scene, message)

    def set_transform(self, **params: np.ndarray) -> None:
        invalid = params.keys() - set(self.model.transform_keys)
        if invalid:
            raise ValueError(f"Invalid transform parameter(s): {', '.join(sorted(invalid))}.")
        self._update_params(params)
        message = BodyModelsViserTransformMessage(
            name=self.name,
            global_rotation=_f32(self.params["global_rotation"]),
            global_translation=_f32(self.params["global_translation"]),
        )
        state = _runtime.get_state(self.scene)
        state.models[self.name] = dataclasses.replace(
            state.models[self.name],
            global_rotation=message.global_rotation,
            global_translation=message.global_translation,
        )
        _runtime.broadcast(self.scene, message)

    def remove(self) -> None:
        state = _runtime.get_state(self.scene)
        del state.models[self.name]
        _release_asset(state, self._asset_key)
        _runtime.broadcast(self.scene, _messages.RemoveSceneNodeMessage(self.name))

    def _prepare_identity(self) -> Any:
        identity_params = {key: self.params[key] for key in self.model.identity_keys}
        return self.model.prepare_identity(**identity_params)

    def _prepare_pose(self) -> Any:
        pose_params = {key: self.params[key] for key in self.model.pose_keys}
        return self.model.prepare_pose(
            **pose_params,
            identity=self._prepared_identity,
            # Models with an explicit corrective basis can skip all per-vertex
            # work. Other model families may need their ordinary preparation
            # path to produce skinning transforms at all.
            skip_vertices=hasattr(self.model, "posedirs"),
        )

    def _prepare_skinning(self) -> tuple[Any, np.ndarray | None]:
        prepared_pose = self._prepare_pose()
        skinning = self.model.prepare_skinning(
            identity=self._prepared_identity,
            pose=prepared_pose,
        )
        if "pose_offsets" in skinning:
            raise ValueError(
                f"{type(self.model).__name__} only exposes server-side pose offsets, which "
                "body-models-viser does not transmit. The model must expose a client-evaluable "
                "posedirs basis or skinning transforms without pose offsets."
            )
        coefficients = None
        if self.use_pose_correctives:
            coefficients = _pose_coefficients(self.model, prepared_pose)
        return skinning, coefficients

    def _update_params(self, params: dict[str, np.ndarray]) -> None:
        for key, value in params.items():
            self.params[key] = np.asarray(value, dtype=np.float32).copy()


def add_body_model(
    scene: Any,
    name: str,
    model: body_models.SkinnedModel,
    *,
    use_pose_correctives: bool = False,
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
    """Add a browser-skinned body model.

    Pose correctives are disabled by default. When enabled, the static
    corrective basis is sent to the browser once and evaluated there; the
    server never computes per-vertex corrective offsets. The basis is
    quantized to signed 16-bit values with one scale per vertex coordinate.
    """
    if not isinstance(model, body_models.SkinnedModel):
        raise TypeError(f"Expected body_models.SkinnedModel, got {type(model).__name__}.")
    if isinstance(model, _SERVER_ONLY_CORRECTIVE_MODELS):
        raise ValueError(
            f"{type(model).__name__} only exposes server-side pose offsets, which "
            "body-models-viser does not compute or transmit."
        )
    if use_pose_correctives and _client_corrective_basis(model) is None:
        raise ValueError(
            f"{type(model).__name__} does not expose compatible client-side pose correctives."
        )
    state = _runtime.get_state(scene)
    if name in state.models:
        raise ValueError(f"A body model named {name!r} already exists.")

    rest_pose = model.get_rest_pose()
    params = {key: np.asarray(value, dtype=np.float32).copy() for key, value in rest_pose.items()}
    handle = BodyModelHandle(
        scene,
        name,
        model,
        params,
        use_pose_correctives=use_pose_correctives,
    )
    skinning, coefficients = handle._prepare_skinning()
    asset, is_new_asset = _acquire_asset(
        state,
        model,
        skinning,
        use_pose_correctives=use_pose_correctives,
    )
    if is_new_asset:
        _runtime.broadcast(scene, asset)

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
        asset_id=asset.asset_id,
        rest_vertices=_f32(skinning["rest_vertices"]),
        skinning_transforms=_f32(skinning["skinning_transforms"]),
        pose_coefficients=coefficients,
        global_rotation=_f32(params["global_rotation"]),
        global_translation=_f32(params["global_translation"]),
        props=props,
    )
    state.models[name] = message
    _runtime.broadcast(scene, message)
    return handle


def _acquire_asset(
    state: _runtime.RuntimeState,
    model: body_models.SkinnedModel,
    skinning: Any,
    *,
    use_pose_correctives: bool,
) -> tuple[BodyModelsViserAssetMessage, bool]:
    # Topology and skin weights are model-static. Keeping the model in the key
    # also prevents Python object-id reuse from aliasing unrelated assets.
    key = (model, use_pose_correctives)
    existing = state.assets.get(key)
    if existing is not None:
        existing.refcount += 1
        return existing.message, False

    asset_id = state.next_asset_id
    state.next_asset_id += 1
    offsets, indices, values = _sparse_skin_weights(skinning["skin_weights"])
    basis = scales = None
    if use_pose_correctives:
        basis, scales = _quantize_corrective_basis(model, skinning["rest_vertices"])
    message = BodyModelsViserAssetMessage(
        asset_id=asset_id,
        faces=np.ascontiguousarray(skinning["faces"], dtype="<u4"),
        skin_weight_offsets=offsets,
        skin_weight_indices=indices,
        skin_weight_values=values,
        corrective_basis=basis,
        corrective_scales=scales,
    )
    state.assets[key] = _runtime._AssetRecord(message)
    return message, True


def _release_asset(
    state: _runtime.RuntimeState,
    key: tuple[body_models.SkinnedModel, bool],
) -> None:
    asset = state.assets[key]
    asset.refcount -= 1
    if asset.refcount:
        return
    del state.assets[key]


def _client_corrective_basis(model: body_models.SkinnedModel) -> np.ndarray | None:
    if isinstance(model, _NONSTANDARD_CORRECTIVE_MODELS):
        return None
    posedirs = getattr(model, "posedirs", None)
    parents = getattr(model, "parents", None)
    if posedirs is None or parents is None:
        return None
    basis = np.asarray(posedirs)
    expected_features = 9 * (len(parents) - 1)
    if basis.ndim != 2 or basis.shape[0] != expected_features:
        return None
    return basis


def _pose_coefficients(model: body_models.SkinnedModel, prepared_pose: Any) -> np.ndarray:
    try:
        world_rotations = np.asarray(prepared_pose["skeleton_transforms"], dtype=np.float32)[
            :, :3, :3
        ]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(
            f"{type(model).__name__} does not expose skeleton_transforms needed for pose correctives."
        ) from exc
    parents = np.asarray(model.parents, dtype=np.int64)
    if world_rotations.shape != (len(parents), 3, 3):
        raise ValueError("Client-side pose correctives currently require an unbatched skeleton.")

    local_rotations = world_rotations.copy()
    for joint in range(1, len(parents)):
        local_rotations[joint] = world_rotations[parents[joint]].T @ world_rotations[joint]
    identity = np.eye(3, dtype=np.float32)
    coefficients = (local_rotations[1:] - identity).reshape(-1)
    basis = _client_corrective_basis(model)
    assert basis is not None
    expected = basis.shape[0]
    if coefficients.size != expected:
        raise ValueError(
            f"Corrective basis expects {expected} pose coefficients, but the skeleton provides "
            f"{coefficients.size}."
        )
    return _f32(coefficients)


def _quantize_corrective_basis(
    model: body_models.SkinnedModel,
    rest_vertices: Any,
) -> tuple[np.ndarray, np.ndarray]:
    posedirs = _client_corrective_basis(model)
    if posedirs is None:
        raise ValueError(f"{type(model).__name__} has no compatible corrective basis.")
    posedirs = np.asarray(posedirs, dtype=np.float32)
    coordinate_count = np.asarray(rest_vertices).size
    if posedirs.ndim != 2 or posedirs.shape[1] != coordinate_count:
        raise ValueError(
            f"Expected posedirs shape [P, {coordinate_count}], got {posedirs.shape}."
        )
    basis = posedirs.T
    scales = np.max(np.abs(basis), axis=1) / 32767.0
    nonzero_scales = np.where(scales == 0.0, 1.0, scales)
    quantized = np.rint(basis / nonzero_scales[:, None]).clip(-32767, 32767)
    return (
        np.ascontiguousarray(quantized, dtype="<i2"),
        np.ascontiguousarray(scales, dtype="<f4"),
    )


def _sparse_skin_weights(weights: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense = np.asarray(weights, dtype=np.float32)
    if dense.ndim != 2:
        raise ValueError(f"Expected skin weights with shape [vertices, joints], got {dense.shape}.")
    if dense.shape[1] > np.iinfo(np.uint16).max:
        raise ValueError("Skinning supports at most 65535 joints.")
    active = dense != 0.0
    counts = active.sum(axis=1, dtype=np.uint32)
    offsets = np.empty(dense.shape[0] + 1, dtype=np.uint32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return (
        np.ascontiguousarray(offsets, dtype="<u4"),
        np.ascontiguousarray(np.nonzero(active)[1], dtype="<u2"),
        np.ascontiguousarray(dense[active], dtype="<f4"),
    )


def _f32(array: Any) -> np.ndarray:
    return np.ascontiguousarray(array, dtype="<f4")
