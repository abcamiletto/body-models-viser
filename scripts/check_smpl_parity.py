# /// script
# dependencies = [
#   "body-models",
#   "wasmtime",
# ]
# [tool.uv.sources]
# body-models = { path = "../../body-models", editable = true }
# ///

from __future__ import annotations

from pathlib import Path

import numpy as np
import wasmtime
from body_models.smpl.numpy import SMPL


def main() -> None:
    model = SMPL(gender="neutral")
    params = model.get_rest_pose()
    identity = model.prepare_identity(params["shape"])
    pose = model.prepare_pose(params["body_pose"], params["pelvis_rotation"], identity=identity)
    expected = model.forward_vertices(
        params["body_pose"],
        params["pelvis_rotation"],
        params["global_rotation"],
        params["global_translation"],
        identity=identity,
    )

    store = wasmtime.Store()
    wasm_path = Path(__file__).parents[1] / "body_models_viser" / "client" / "body-models-viser.wasm"
    instance = wasmtime.Instance(store, wasmtime.Module.from_file(store.engine, wasm_path), [])
    exports = instance.exports(store)
    memory = exports["memory"]
    alloc = exports["alloc"]
    forward = exports["smpl_forward_vertices"]

    lbs_weights = write_f32(store, memory, alloc, model.weights.lbs_weights)
    rest_joints = write_f32(store, memory, alloc, identity["rest_joints"])
    rest_vertices = write_f32(store, memory, alloc, identity["rest_vertices"])
    joint_transforms = write_f32(store, memory, alloc, pose["joint_transforms"])
    pose_offsets = write_f32(store, memory, alloc, pose["pose_offsets"])
    global_rotation = write_f32(store, memory, alloc, params["global_rotation"])
    global_translation = write_f32(store, memory, alloc, params["global_translation"])
    output = alloc(store, expected.size * 4)

    forward(
        store,
        lbs_weights,
        model.weights.lbs_weights.size,
        rest_joints,
        identity["rest_joints"].size,
        rest_vertices,
        identity["rest_vertices"].size,
        joint_transforms,
        pose["joint_transforms"].size,
        pose_offsets,
        pose["pose_offsets"].size,
        global_rotation,
        global_translation,
        output,
    )

    output_bytes = memory.read(store, output, output + expected.size * 4)
    actual = np.frombuffer(output_bytes, dtype="<f4").reshape(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def write_f32(store: wasmtime.Store, memory: wasmtime.Memory, alloc: wasmtime.Func, values: np.ndarray) -> int:
    array = np.ascontiguousarray(values, dtype="<f4").ravel()
    ptr = alloc(store, array.nbytes)
    memory.write(store, array.tobytes(), ptr)
    return ptr


if __name__ == "__main__":
    main()
