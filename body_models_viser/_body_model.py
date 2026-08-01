from __future__ import annotations

import dataclasses
from typing import Any

import body_models
import numpy as np
from viser import _messages

from . import _runtime
from ._runtime import (
    BodyModelsViserAssetMessage,
    BodyModelsViserIdentityMessage,
    BodyModelsViserModelMessage,
    BodyModelsViserPoseMessage,
    BodyModelsViserTransformMessage,
)

Params = dict[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class _DenseCorrectives:
    values: np.ndarray
    scales: np.ndarray


@dataclasses.dataclass(frozen=True)
class _SparseCorrectives:
    offsets: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    scales: np.ndarray


_QuantizedCorrectives = _DenseCorrectives | _SparseCorrectives


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
            spec = model.parameter_spec.get(key)
            if spec is not None and spec.role == "identity":
                self.set_identity(**{key: value})
                return
            if spec is not None and spec.role == "pose":
                self.set_pose(**{key: value})
                return
            if spec is not None and spec.role == "transform":
                self.set_transform(**{key: value})
                return
        super().__setattr__(key, value)

    def set_identity(self, **params: np.ndarray) -> None:
        invalid = params.keys() - _parameter_keys(self.model, "identity")
        if invalid:
            names = ", ".join(sorted(invalid))
            raise ValueError(f"Invalid identity parameter(s): {names}.")
        self._update_params(params)
        self._prepared_identity = self._prepare_identity()
        pose, coefficients = self._prepare_deformation()
        message = BodyModelsViserIdentityMessage(
            name=self.name,
            rest_vertices=_f32(self._prepared_identity["rest_vertices"]),
            skinning_transforms=_f32(pose["skinning_transforms"]),
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
        invalid = params.keys() - _parameter_keys(self.model, "pose")
        if invalid:
            names = ", ".join(sorted(invalid))
            raise ValueError(f"Invalid pose parameter(s): {names}.")
        self._update_params(params)
        pose, coefficients = self._prepare_deformation()
        message = BodyModelsViserPoseMessage(
            name=self.name,
            skinning_transforms=_f32(pose["skinning_transforms"]),
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
        invalid = params.keys() - _parameter_keys(self.model, "transform")
        if invalid:
            names = ", ".join(sorted(invalid))
            raise ValueError(f"Invalid transform parameter(s): {names}.")
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
        identity_keys = _parameter_keys(self.model, "identity")
        identity_params = {key: self.params[key] for key in identity_keys}
        identity = self.model.prepare_identity(**identity_params)
        if identity["rest_vertices"].ndim != 2:
            raise ValueError("body-models-viser renders one model instance per handle.")
        return identity

    def _prepare_pose(self) -> body_models.SkinningPose:
        pose_keys = _parameter_keys(self.model, "pose")
        pose_params = {key: self.params[key] for key in pose_keys}
        return self.model.prepare_pose(**pose_params, identity=self._prepared_identity)

    def _prepare_deformation(
        self,
    ) -> tuple[body_models.SkinningPose, np.ndarray | None]:
        pose = self._prepare_pose()
        if pose["skinning_transforms"].ndim != 3:
            raise ValueError("body-models-viser renders one model instance per handle.")
        coefficients = None
        if self.use_pose_correctives:
            coefficients = _f32(pose["pose_coefficients"])
            basis = self.model.skinning_spec.corrective_basis
            if coefficients.shape != (basis.coefficient_dim,):
                raise ValueError(
                    f"Expected {basis.coefficient_dim} pose coefficients, "
                    f"got shape {coefficients.shape}."
                )
        return pose, coefficients

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

    Pose correctives are disabled by default. When enabled, the model's dense
    or sparse corrective basis is quantized once and evaluated in the browser.
    """
    if not isinstance(model, body_models.SkinnedModel):
        model_name = type(model).__name__
        raise TypeError(f"Expected body_models.SkinnedModel, got {model_name}.")
    state = _runtime.get_state(scene)
    if name in state.models:
        raise ValueError(f"A body model named {name!r} already exists.")

    spec = model.skinning_spec
    corrective_basis = spec.corrective_basis if use_pose_correctives else None
    if use_pose_correctives and corrective_basis is None:
        model_name = type(model).__name__
        raise ValueError(f"{model_name} has no pose-corrective basis.")
    rest_pose = model.get_rest_pose()
    params = {
        key: np.asarray(value, dtype=np.float32).copy()
        for key, value in rest_pose.items()
    }
    handle = BodyModelHandle(
        scene,
        name,
        model,
        params,
        use_pose_correctives=use_pose_correctives,
    )
    pose, coefficients = handle._prepare_deformation()
    asset, is_new_asset = _acquire_asset(
        state,
        model,
        spec,
        corrective_basis,
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
        rest_vertices=_f32(handle._prepared_identity["rest_vertices"]),
        skinning_transforms=_f32(pose["skinning_transforms"]),
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
    spec: body_models.SkinningSpec,
    corrective_basis: body_models.CorrectiveBasis | None,
) -> tuple[BodyModelsViserAssetMessage, bool]:
    # Topology and skin weights are model-static. Keeping the model in the key
    # also prevents Python object-id reuse from aliasing unrelated assets.
    key = (model, corrective_basis is not None)
    existing = state.assets.get(key)
    if existing is not None:
        existing.refcount += 1
        return existing.message, False

    asset_id = state.next_asset_id
    state.next_asset_id += 1
    offsets, indices, values = _sparse_skin_weights(spec.skinning_weights)
    correctives = (
        None
        if corrective_basis is None
        else _quantize_corrective_basis(corrective_basis)
    )
    if correctives is None:
        corrective_format = None
        corrective_values = corrective_scales = None
        corrective_offsets = corrective_indices = None
    elif isinstance(correctives, _DenseCorrectives):
        corrective_format = "dense"
        corrective_values = correctives.values
        corrective_scales = correctives.scales
        corrective_offsets = corrective_indices = None
    else:
        corrective_format = "sparse"
        corrective_values = correctives.values
        corrective_scales = correctives.scales
        corrective_offsets = correctives.offsets
        corrective_indices = correctives.indices
    message = BodyModelsViserAssetMessage(
        asset_id=asset_id,
        faces=np.ascontiguousarray(spec.triangles, dtype="<u4"),
        skin_weight_offsets=offsets,
        skin_weight_indices=indices,
        skin_weight_values=values,
        corrective_format=corrective_format,
        corrective_values=corrective_values,
        corrective_scales=corrective_scales,
        corrective_offsets=corrective_offsets,
        corrective_indices=corrective_indices,
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


def _quantize_corrective_basis(
    corrective_basis: body_models.CorrectiveBasis,
) -> _QuantizedCorrectives:
    if isinstance(corrective_basis, body_models.DenseCorrectiveBasis):
        values = np.asarray(corrective_basis.values, dtype=np.float32).T
        quantized, scales = _quantize_rows(values)
        return _DenseCorrectives(quantized.ravel(), scales)

    coo = corrective_basis.to_coo()
    if coo.shape[0] > np.iinfo(np.uint16).max + 1:
        raise ValueError("Sparse correctives support at most 65536 coefficients.")
    coefficient_indices = np.asarray(coo.row_indices)
    coordinate_indices = np.asarray(coo.column_indices)
    order = np.argsort(coordinate_indices, kind="stable")
    coordinate_indices = coordinate_indices[order]
    coefficient_indices = coefficient_indices[order]
    values = np.asarray(coo.values, dtype=np.float32)[order]

    coordinate_count = coo.shape[1]
    counts = np.bincount(coordinate_indices, minlength=coordinate_count)
    offsets = np.empty(coordinate_count + 1, dtype="<u4")
    offsets[0] = 0
    np.cumsum(counts, dtype=np.uint32, out=offsets[1:])
    maxima = np.zeros(coordinate_count, dtype=np.float32)
    np.maximum.at(maxima, coordinate_indices, np.abs(values))
    scales = maxima / 32767.0
    entry_scales = scales[coordinate_indices]
    denominators = np.where(entry_scales == 0.0, 1.0, entry_scales)
    quantized = np.rint(values / denominators).clip(-32767, 32767)
    return _SparseCorrectives(
        offsets=offsets,
        indices=np.ascontiguousarray(coefficient_indices, dtype="<u2"),
        values=np.ascontiguousarray(quantized, dtype="<i2"),
        scales=np.ascontiguousarray(scales, dtype="<f4"),
    )


def _quantize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maxima = np.maximum(values.max(axis=1), -values.min(axis=1))
    scales = maxima / 32767.0
    nonzero_scales = np.where(scales == 0.0, 1.0, scales)
    quantized = np.empty_like(values)
    np.divide(values, nonzero_scales[:, None], out=quantized)
    np.rint(quantized, out=quantized)
    np.clip(quantized, -32767, 32767, out=quantized)
    return (
        np.ascontiguousarray(quantized, dtype="<i2"),
        np.ascontiguousarray(scales, dtype="<f4"),
    )


def _sparse_skin_weights(weights: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense = np.asarray(weights, dtype=np.float32)
    if dense.ndim != 2:
        raise ValueError(
            f"Expected skin weights with shape [vertices, joints], got {dense.shape}."
        )
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


def _parameter_keys(
    model: body_models.ArticulatedModel,
    role: body_models.ParameterRole,
) -> set[str]:
    return {name for name, spec in model.parameter_spec.items() if spec.role == role}
