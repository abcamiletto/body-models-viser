from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import wasmtime
from body_models.anny.numpy import ANNY
from body_models.flame.numpy import FLAME
from body_models.garment_measurements.numpy import GarmentMeasurements
from body_models.mano.numpy import MANO
from body_models.mhr.numpy import MHR
from body_models.skel.numpy import SKEL
from body_models.smpl.numpy import SMPL
from body_models.smplh.numpy import SMPLH
from body_models.smplx.numpy import SMPLX
from body_models.soma.numpy import SOMA

from body_models_viser._body_model import (
    _DenseCorrectives,
    _quantize_corrective_basis,
    _sparse_skin_weights,
    _SparseCorrectives,
)


def main() -> None:
    store = wasmtime.Store()
    wasm_path = (
        Path(__file__).parents[1] / "body_models_viser/client/body-models-viser.wasm"
    )
    instance = wasmtime.Instance(
        store,
        wasmtime.Module.from_file(store.engine, wasm_path),
        [],
    )
    exports = instance.exports(store)
    memory = exports["memory"]
    alloc = exports["alloc"]
    compute_dense_offsets = exports["compute_dense_pose_offsets"]
    compute_sparse_offsets = exports["compute_sparse_pose_offsets"]
    forward = exports["forward_vertices_sparse"]

    for name, make_model in MODELS:
        model = make_model()
        params = {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in model.get_rest_pose().items()
        }
        pose_keys = [
            key for key, spec in model.parameter_spec.items() if spec.role == "pose"
        ]
        for key in pose_keys:
            if params[key].size:
                params[key].flat[0] = 0.15
        params["global_rotation"] = np.array([0.2, -0.1, 0.15], dtype=np.float32)
        params["global_translation"] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        identity_params = {
            key: params[key]
            for key, spec in model.parameter_spec.items()
            if spec.role == "identity"
        }
        identity = model.prepare_identity(**identity_params)
        pose_params = {
            key: params[key]
            for key, spec in model.parameter_spec.items()
            if spec.role == "pose"
        }
        pose = model.prepare_pose(**pose_params, identity=identity)
        spec = model.skinning_spec
        rest_vertices = identity["rest_vertices"]
        transforms = pose["skinning_transforms"]
        offsets, indices, values = _sparse_skin_weights(spec.skinning_weights)
        pose_offsets_array = np.zeros_like(rest_vertices)
        rtol = atol = 1e-5

        if spec.corrective_basis is not None:
            correctives = _quantize_corrective_basis(spec.corrective_basis)
            coefficients = pose["pose_coefficients"]
            values_ptr = write_array(store, memory, alloc, correctives.values)
            scales_ptr = write_array(store, memory, alloc, correctives.scales)
            coefficients_ptr = write_array(store, memory, alloc, coefficients)
            pose_offsets_ptr = write_array(store, memory, alloc, pose_offsets_array)
            if isinstance(correctives, _DenseCorrectives):
                compute_dense_offsets(
                    store,
                    values_ptr,
                    correctives.values.size,
                    scales_ptr,
                    correctives.scales.size,
                    coefficients_ptr,
                    coefficients.size,
                    pose_offsets_ptr,
                    pose_offsets_array.size,
                )
            elif isinstance(correctives, _SparseCorrectives):
                rtol, atol = 2e-3, 2e-4
                offsets_ptr = write_array(store, memory, alloc, correctives.offsets)
                indices_ptr = write_array(store, memory, alloc, correctives.indices)
                compute_sparse_offsets(
                    store,
                    offsets_ptr,
                    correctives.offsets.size,
                    indices_ptr,
                    correctives.indices.size,
                    values_ptr,
                    correctives.values.size,
                    scales_ptr,
                    correctives.scales.size,
                    coefficients_ptr,
                    coefficients.size,
                    pose_offsets_ptr,
                    pose_offsets_array.size,
                )
            pose_offsets_array = read_f32(
                store,
                memory,
                pose_offsets_ptr,
                pose_offsets_array.shape,
            )
        else:
            pose_offsets_ptr = write_array(store, memory, alloc, pose_offsets_array)

        forward_params = {
            key: value
            for key, value in params.items()
            if model.parameter_spec[key].role != "identity"
        }
        expected = model.forward_vertices(**forward_params, identity=identity)
        skin_offsets_ptr = write_array(store, memory, alloc, offsets)
        skin_indices_ptr = write_array(store, memory, alloc, indices)
        skin_values_ptr = write_array(store, memory, alloc, values)
        rest_ptr = write_array(store, memory, alloc, rest_vertices)
        transforms_ptr = write_array(store, memory, alloc, transforms)
        rotation_ptr = write_array(store, memory, alloc, params["global_rotation"])
        translation_ptr = write_array(
            store, memory, alloc, params["global_translation"]
        )
        output_ptr = alloc(store, expected.size * 4)
        forward(
            store,
            skin_offsets_ptr,
            offsets.size,
            skin_indices_ptr,
            indices.size,
            skin_values_ptr,
            values.size,
            rest_ptr,
            rest_vertices.size,
            transforms_ptr,
            transforms.size,
            pose_offsets_ptr,
            pose_offsets_array.size,
            rotation_ptr,
            translation_ptr,
            output_ptr,
        )
        actual = read_f32(store, memory, output_ptr, expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
        print(f"{name}: ok")


def write_array(store, memory, alloc, values) -> int:
    array = np.ascontiguousarray(values)
    ptr = alloc(store, array.nbytes)
    memory.write(store, array.tobytes(), ptr)
    return ptr


def read_f32(store, memory, ptr: int, shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape))
    buffer = memory.read(store, ptr, ptr + size * 4)
    return np.frombuffer(buffer, dtype="<f4").reshape(shape)


MODELS: list[tuple[str, Callable[[], Any]]] = [
    ("ANNY", ANNY),
    ("FLAME", FLAME),
    ("GarmentMeasurements", GarmentMeasurements),
    ("MANO", lambda: MANO(side="right")),
    ("MHR", MHR),
    ("SKEL", lambda: SKEL(gender="male")),
    ("SMPL", lambda: SMPL(gender="neutral")),
    ("SMPLH", lambda: SMPLH(gender="neutral")),
    ("SMPLX", lambda: SMPLX(gender="neutral")),
    ("SOMA", SOMA),
]


if __name__ == "__main__":
    main()
