from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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
from body_models_viser import BodyModelHandle
from body_models_viser._body_model import (
    _client_corrective_basis,
    _pose_coefficients,
    _quantize_corrective_basis,
    _sparse_skin_weights,
)


def main() -> None:
    store = wasmtime.Store()
    wasm_path = Path(__file__).parents[1] / "body_models_viser/client/body-models-viser.wasm"
    instance = wasmtime.Instance(
        store,
        wasmtime.Module.from_file(store.engine, wasm_path),
        [],
    )
    exports = instance.exports(store)
    memory = exports["memory"]
    alloc = exports["alloc"]
    compute_offsets = exports["compute_pose_offsets"]
    forward = exports["forward_vertices_sparse"]

    for name, make_model in MODELS:
        model = make_model()
        params = {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in model.get_rest_pose().items()
        }
        for key in model.pose_keys:
            if params[key].size:
                params[key].flat[0] = 0.15
        params["global_rotation"] = np.array([0.2, -0.1, 0.15], dtype=np.float32)
        params["global_translation"] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        handle = BodyModelHandle(
            scene=None,
            name=name,
            model=model,
            params=params,
            use_pose_correctives=False,
        )
        pose = handle._prepare_pose()
        skinning = model.prepare_skinning(identity=handle._prepared_identity, pose=pose)
        if "pose_offsets" in skinning and not hasattr(model, "posedirs"):
            print(f"{name}: unsupported server-only pose offsets")
            continue
        corrective_basis = _client_corrective_basis(model)
        if hasattr(model, "posedirs") and corrective_basis is None:
            print(f"{name}: unsupported corrective feature mapping")
            continue
        offsets, indices, values = _sparse_skin_weights(skinning["skin_weights"])
        pose_offsets_array = np.zeros_like(skinning["rest_vertices"])

        if corrective_basis is not None:
            basis, scales = _quantize_corrective_basis(model, skinning["rest_vertices"])
            coefficients = _pose_coefficients(model, pose)
            basis_ptr = write_array(store, memory, alloc, basis)
            scales_ptr = write_array(store, memory, alloc, scales)
            coefficients_ptr = write_array(store, memory, alloc, coefficients)
            pose_offsets_ptr = write_array(store, memory, alloc, pose_offsets_array)
            compute_offsets(
                store,
                basis_ptr,
                basis.size,
                scales_ptr,
                scales.size,
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

        expected = model.forward_vertices(**params, identity=handle._prepared_identity)
        offsets_ptr = write_array(store, memory, alloc, offsets)
        indices_ptr = write_array(store, memory, alloc, indices)
        values_ptr = write_array(store, memory, alloc, values)
        rest_ptr = write_array(store, memory, alloc, skinning["rest_vertices"])
        transforms_ptr = write_array(store, memory, alloc, skinning["skinning_transforms"])
        rotation_ptr = write_array(store, memory, alloc, params["global_rotation"])
        translation_ptr = write_array(store, memory, alloc, params["global_translation"])
        output_ptr = alloc(store, expected.size * 4)
        forward(
            store,
            offsets_ptr,
            offsets.size,
            indices_ptr,
            indices.size,
            values_ptr,
            values.size,
            rest_ptr,
            skinning["rest_vertices"].size,
            transforms_ptr,
            skinning["skinning_transforms"].size,
            pose_offsets_ptr,
            pose_offsets_array.size,
            rotation_ptr,
            translation_ptr,
            output_ptr,
        )
        actual = read_f32(store, memory, output_ptr, expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, err_msg=name)
        print(f"{name}: ok")


def write_array(store, memory, alloc, values) -> int:
    array = np.ascontiguousarray(values)
    ptr = alloc(store, array.nbytes)
    memory.write(store, array.tobytes(), ptr)
    return ptr


def read_f32(store, memory, ptr: int, shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape))
    return np.frombuffer(memory.read(store, ptr, ptr + size * 4), dtype="<f4").reshape(shape)


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
