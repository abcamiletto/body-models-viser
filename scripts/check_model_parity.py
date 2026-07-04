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
from body_models_viser._body_model import _HANDLE_TYPES


def main() -> None:
    store = wasmtime.Store()
    client_dir = Path(__file__).parents[1] / "body_models_viser" / "client"
    wasm_path = client_dir / "body-models-viser.wasm"
    module = wasmtime.Module.from_file(store.engine, wasm_path)
    instance = wasmtime.Instance(store, module, [])
    exports = instance.exports(store)
    memory = exports["memory"]
    alloc = exports["alloc"]
    forward = exports["forward_vertices"]

    for name, make_model in MODELS:
        model = make_model()
        rest_pose = model.get_rest_pose()
        params = {key: np.asarray(value, dtype=np.float32).copy() for key, value in rest_pose.items()}
        params["global_rotation"] = np.array([0.2, -0.1, 0.15], dtype=np.float32)
        params["global_translation"] = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        handle = _HANDLE_TYPES[type(model)](scene=None, name=name, model=model, params=params)
        identity = handle._prepared_identity
        skinning = model.prepare_skinning(identity=identity, pose=handle._prepare_pose())
        pose_offsets_array = skinning["pose_offsets"] if "pose_offsets" in skinning else np.zeros_like(skinning["rest_vertices"])
        expected = model.forward_vertices(**params, identity=identity)

        skin_weights = write_f32(store, memory, alloc, skinning["skin_weights"])
        rest_vertices = write_f32(store, memory, alloc, skinning["rest_vertices"])
        skinning_transforms = write_f32(store, memory, alloc, skinning["skinning_transforms"])
        pose_offsets = write_f32(store, memory, alloc, pose_offsets_array)
        global_rotation = write_f32(store, memory, alloc, params["global_rotation"])
        global_translation = write_f32(store, memory, alloc, params["global_translation"])
        output = alloc(store, expected.size * 4)

        forward(
            store,
            skin_weights,
            skinning["skin_weights"].size,
            rest_vertices,
            skinning["rest_vertices"].size,
            skinning_transforms,
            skinning["skinning_transforms"].size,
            pose_offsets,
            pose_offsets_array.size,
            global_rotation,
            global_translation,
            output,
        )

        output_bytes = memory.read(store, output, output + expected.size * 4)
        actual = np.frombuffer(output_bytes, dtype="<f4").reshape(expected.shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, err_msg=name)
        print(f"{name}: ok")


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


def write_f32(store: wasmtime.Store, memory: wasmtime.Memory, alloc: wasmtime.Func, values: np.ndarray) -> int:
    array = np.ascontiguousarray(values, dtype="<f4").ravel()
    ptr = alloc(store, array.nbytes)
    memory.write(store, array.tobytes(), ptr)
    return ptr


if __name__ == "__main__":
    main()
