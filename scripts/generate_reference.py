from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from body_models.mhr.torch import MHR
from body_models.smpl.torch import SMPL


def tensor1(values: list[float], size: int, name: str) -> torch.Tensor:
    if len(values) != size:
        raise ValueError(f"Expected {name} to have length {size}, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32)


def tensor2(values: list[list[float]], rows: int, cols: int, name: str) -> torch.Tensor:
    if len(values) != rows or any(len(row) != cols for row in values):
        raise ValueError(f"Expected {name} to have shape [{rows}, {cols}]")
    return torch.tensor(values, dtype=torch.float32)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def smpl_params(model: SMPL, fixture: dict[str, Any]) -> dict[str, torch.Tensor]:
    params = fixture["params"]
    return {
        "shape": tensor1(params["shape"], 10, "SMPL shape")[None],
        "body_pose": tensor2(params["body_pose"], model.NUM_BODY_JOINTS, 3, "SMPL body_pose")[None],
        "pelvis_rotation": tensor1(params["pelvis_rotation"], 3, "SMPL pelvis_rotation")[None],
        "global_rotation": tensor1(params["global_rotation"], 3, "SMPL global_rotation")[None],
        "global_translation": tensor1(params["global_translation"], 3, "SMPL global_translation")[None],
    }


def mhr_params(model: MHR, fixture: dict[str, Any]) -> dict[str, torch.Tensor]:
    params = fixture["params"]
    return {
        "shape": tensor1(params["shape"], model.SHAPE_DIM, "MHR shape")[None],
        "body_pose": tensor1(params["body_pose"], model.body_pose_dim, "MHR body_pose")[None],
        "hand_pose": tensor1(params["hand_pose"], model.hand_pose_dim, "MHR hand_pose")[None],
        "expression": tensor1(params["expression"], model.EXPR_DIM, "MHR expression")[None],
        "global_rotation": tensor1(params["global_rotation"], 3, "MHR global_rotation")[None],
        "global_translation": tensor1(params["global_translation"], 3, "MHR global_translation")[None],
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), separators=(",", ":")), encoding="utf-8")


def export_smpl(root: Path, out: Path) -> None:
    model = SMPL(gender="neutral").eval()
    weights = model.weights
    write_json(
        out / "model_data" / "smpl.json",
        {
            "v_template": weights.v_template,
            "faces": weights.faces,
            "lbs_weights": weights.lbs_weights,
            "shapedirs": weights.shapedirs[:, :, :10],
            "posedirs": weights.posedirs,
            "j_template": weights.j_template,
            "j_shapedirs": weights.j_shapedirs[:, :, :10],
            "parents": weights.parents,
        },
    )
    for fixture_path in sorted((root / "fixtures" / "smpl").glob("*.json")):
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        params = smpl_params(model, fixture)
        write_json(
            out / "reference" / "smpl" / fixture_path.name,
            {
                "model": "smpl",
                "case": fixture["case"],
                "skeleton": model.forward_skeleton(**params)[0],
                "mesh": model.forward_vertices(**params)[0],
            },
        )


def export_mhr(root: Path, out: Path) -> None:
    model = MHR().eval()
    weights = model.weights
    write_json(
        out / "model_data" / "mhr.json",
        {
            "base_vertices": weights.base_vertices,
            "blendshape_dirs": weights.blendshape_dirs,
            "skin_weights": weights.skin_weights,
            "skin_indices": weights.skin_indices,
            "faces": weights.faces,
            "joint_offsets": weights.joint_offsets,
            "joint_pre_rotations": weights.joint_pre_rotations,
            "parameter_transform": weights.parameter_transform,
            "bind_inv_linear": weights.bind_inv_linear,
            "bind_inv_translation": weights.bind_inv_translation,
            "corrective_W1": weights.corrective_W1,
            "corrective_W2": weights.corrective_W2,
            "parents": weights.parents,
        },
    )
    for fixture_path in sorted((root / "fixtures" / "mhr").glob("*.json")):
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        params = mhr_params(model, fixture)
        write_json(
            out / "reference" / "mhr" / fixture_path.name,
            {
                "model": "mhr",
                "case": fixture["case"],
                "skeleton": model.forward_skeleton(**params)[0],
                "mesh": model.forward_vertices(**params)[0],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("generated"))
    parser.add_argument("--models", nargs="+", choices=("smpl", "mhr"), default=("smpl", "mhr"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = args.output if args.output.is_absolute() else root / args.output
    if "smpl" in args.models:
        export_smpl(root, out)
    if "mhr" in args.models:
        export_mhr(root, out)


if __name__ == "__main__":
    main()
