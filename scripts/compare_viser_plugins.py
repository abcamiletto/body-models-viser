from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import viser
from nanomanifold import SO3

from body_models.anny.torch import ANNY
from body_models.extras.viser_plugin import add_body_model
from body_models.garment_measurements.torch import GarmentMeasurements
from body_models.mhr.torch import MHR
from body_models.smpl.torch import SMPL
from body_models.soma.torch import SOMA


ROOT = Path(__file__).resolve().parents[1]

PYTHON_MODELS = {
    "smpl": lambda: SMPL(gender="neutral").eval(),
    "mhr": lambda: MHR().eval(),
    "anny": lambda: ANNY().eval(),
    "soma": lambda: SOMA().eval(),
    "garment": lambda: GarmentMeasurements().eval(),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("smpl", "mhr", "anny", "soma", "garment", "all"),
        default="all",
    )
    parser.add_argument("--case", default="shape_pose", choices=("rest", "shape_pose", "translation"))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    ensure_generated()
    subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=6.0, height=4.0)

    names = list(PYTHON_MODELS) if args.model == "all" else [args.model]
    for row, model_name in enumerate(names):
        add_comparison(server.scene, model_name, args.case, y=-1.1 * row)

    print(f"Open http://localhost:{args.port}")
    server.sleep_forever()


def add_comparison(scene: viser.SceneApi, model_name: str, case: str, y: float) -> None:
    fixture = load_json(ROOT / "fixtures" / model_name / f"{case}.json")
    model = PYTHON_MODELS[model_name]()

    original = add_body_model(scene, f"/{model_name}/original", model, color=(80, 160, 255))
    original.position = (-0.75, y, 0.0)
    original.set_pose(**PARAMS[model_name](fixture["params"]))

    rust = run_rust_model(model_name, case)
    weights = load_json(ROOT / "generated" / "model_data" / f"{model_name}.json")
    root = scene.add_frame(f"/{model_name}/rust", show_axes=False, position=(0.75, y, 0.0))
    scene.add_mesh_skinned(
        f"{root.name}/mesh",
        vertices=np.asarray(rust["mesh"], dtype=np.float32),
        faces=triangular_faces(np.asarray(weights["faces"], dtype=np.uint32)),
        bone_wxyzs=bone_wxyzs(rust["skeleton"]),
        bone_positions=bone_positions(rust["skeleton"]),
        skin_weights=dense_skin_weights(model_name, weights),
        color=(255, 160, 80),
    )

    scene.add_label(f"/{model_name}/original/label", "original", position=(0.0, 0.0, 1.2))
    scene.add_label(f"/{model_name}/rust/label", "rust", position=(0.0, 0.0, 1.2))


def smpl_params(params: dict[str, Any]) -> dict[str, torch.Tensor]:
    names = ("shape", "body_pose", "pelvis_rotation", "global_rotation", "global_translation")
    return tensors(params, names)


def mhr_params(params: dict[str, Any]) -> dict[str, torch.Tensor]:
    names = ("shape", "body_pose", "hand_pose", "expression", "global_rotation", "global_translation")
    return tensors(params, names)


def anny_params(params: dict[str, Any]) -> dict[str, torch.Tensor]:
    names = (
        "gender",
        "age",
        "muscle",
        "weight",
        "height",
        "proportions",
        "body_pose",
        "head_pose",
        "hand_pose",
        "global_rotation",
        "global_translation",
    )
    return tensors(params, names)


def soma_params(params: dict[str, Any]) -> dict[str, torch.Tensor]:
    names = ("body_pose", "head_pose", "hand_pose", "global_rotation", "global_translation")
    return tensors(params, names)


def garment_params(params: dict[str, Any]) -> dict[str, torch.Tensor]:
    names = (
        "shape",
        "body_pose",
        "head_pose",
        "hand_pose",
        "pelvis_rotation",
        "global_rotation",
        "global_translation",
    )
    return tensors(params, names)


def run_rust_model(model_name: str, case: str) -> dict[str, Any]:
    binary = ROOT / "target" / "release" / "body-models-viser"
    result = subprocess.run(
        [
            str(binary),
            "--model-data",
            str(ROOT / "generated" / "model_data"),
            "--input",
            str(ROOT / "fixtures" / model_name / f"{case}.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def ensure_generated() -> None:
    required = [
        ROOT / "generated" / "model_data" / "smpl.json",
        ROOT / "generated" / "model_data" / "mhr.json",
        ROOT / "generated" / "model_data" / "anny.json",
        ROOT / "generated" / "model_data" / "soma.json",
        ROOT / "generated" / "model_data" / "garment.json",
    ]
    if not all(path.exists() for path in required):
        raise FileNotFoundError("Run scripts/generate_reference.py before launching the comparison viewer.")


def tensors(params: dict[str, Any], names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(params[name], dtype=torch.float32) for name in names}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def triangular_faces(faces: np.ndarray) -> np.ndarray:
    if faces.shape[1] == 3:
        return faces
    if faces.shape[1] == 4:
        return np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
    raise ValueError(f"Expected triangular or quad faces, got {faces.shape}.")


def bone_wxyzs(skeleton: list[Any]) -> np.ndarray:
    rotations = np.asarray(skeleton, dtype=np.float32)[:, :3, :3]
    return SO3.conversions.from_rotmat_to_quat(rotations, convention="wxyz", xp=np)


def bone_positions(skeleton: list[Any]) -> np.ndarray:
    return np.asarray(skeleton, dtype=np.float32)[:, :3, 3]


def dense_skin_weights(model_name: str, weights: dict[str, Any]) -> np.ndarray:
    if model_name == "smpl":
        return np.asarray(weights["lbs_weights"], dtype=np.float32)
    if model_name == "anny":
        return sparse_to_dense(weights["lbs_joint_indices"], weights["lbs_joint_weights"], len(weights["parents"]))
    if model_name == "soma":
        sparse_weights = np.asarray(weights["skin_joint_weights"], dtype=np.float32)
        sparse_indices = np.asarray(weights["skin_joint_indices"], dtype=np.int64)
        dense = np.zeros((sparse_weights.shape[0], len(weights["parents"]) - 1), dtype=np.float32)
        valid = sparse_indices > 0
        rows, slots = np.where(valid)
        dense[rows, sparse_indices[rows, slots] - 1] = sparse_weights[rows, slots]
        return dense
    if model_name == "garment":
        return sparse_to_dense(weights["skin_joint_indices"], weights["skin_joint_weights"], len(weights["parents"]))

    return sparse_to_dense(weights["skin_indices"], weights["skin_weights"], len(weights["parents"]))


def sparse_to_dense(indices: Any, weights: Any, num_joints: int) -> np.ndarray:
    sparse_indices = np.asarray(indices, dtype=np.int64)
    sparse_weights = np.asarray(weights, dtype=np.float32)
    dense = np.zeros((sparse_weights.shape[0], num_joints), dtype=np.float32)
    np.put_along_axis(dense, sparse_indices, sparse_weights, axis=1)
    return dense


PARAMS = {
    "smpl": smpl_params,
    "mhr": mhr_params,
    "anny": anny_params,
    "soma": soma_params,
    "garment": garment_params,
}


if __name__ == "__main__":
    main()
