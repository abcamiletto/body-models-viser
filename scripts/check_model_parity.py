from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import wasmtime
from body_models.anny.numpy import ANNY
from body_models.flame.numpy import FLAME
from body_models.mano.numpy import MANO
from body_models.mhr.numpy import MHR
from body_models.skel.numpy import SKEL
from body_models.smpl.numpy import SMPL
from body_models.smplh.numpy import SMPLH
from body_models.smplx.numpy import SMPLX
from body_models.soma.numpy import SOMA
from body_models_viser._viser import _prepare_identity, _prepare_pose, _runtime_inputs


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
        identity = _prepare_identity(model, params)
        pose = _prepare_pose(model, params, identity)
        runtime_inputs = _runtime_inputs(model, identity, pose)
        expected = model.forward_vertices(**params, identity=identity)

        lbs_weights = write_f32(store, memory, alloc, runtime_inputs.lbs_weights)
        rest_vertices = write_f32(store, memory, alloc, runtime_inputs.rest_vertices)
        skinning_transforms = write_f32(store, memory, alloc, runtime_inputs.skinning_transforms)
        pose_offsets = write_f32(store, memory, alloc, runtime_inputs.pose_offsets)
        global_rotation = write_f32(store, memory, alloc, params["global_rotation"])
        global_translation = write_f32(store, memory, alloc, params["global_translation"])
        output = alloc(store, expected.size * 4)

        forward(
            store,
            lbs_weights,
            runtime_inputs.lbs_weights.size,
            rest_vertices,
            runtime_inputs.rest_vertices.size,
            skinning_transforms,
            runtime_inputs.skinning_transforms.size,
            pose_offsets,
            runtime_inputs.pose_offsets.size,
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
