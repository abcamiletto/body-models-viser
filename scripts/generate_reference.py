from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from body_models.anny.torch import ANNY
from body_models.garment_measurements.torch import GarmentMeasurements
from body_models.mhr.torch import MHR
from body_models.smpl.torch import SMPL
from body_models.soma.pose import pack_pose as pack_soma_pose
from body_models.soma.torch import SOMA


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


def anny_params(model: ANNY, fixture: dict[str, Any]) -> dict[str, torch.Tensor]:
    params = fixture["params"]
    return {
        "gender": tensor1([params["gender"]], 1, "ANNY gender"),
        "age": tensor1([params["age"]], 1, "ANNY age"),
        "muscle": tensor1([params["muscle"]], 1, "ANNY muscle"),
        "weight": tensor1([params["weight"]], 1, "ANNY weight"),
        "height": tensor1([params["height"]], 1, "ANNY height"),
        "proportions": tensor1([params["proportions"]], 1, "ANNY proportions"),
        "body_pose": tensor2(params["body_pose"], 64, 3, "ANNY body_pose")[None],
        "head_pose": tensor2(params["head_pose"], 60, 3, "ANNY head_pose")[None],
        "hand_pose": tensor2(params["hand_pose"], 38, 3, "ANNY hand_pose")[None],
        "global_rotation": tensor1(params["global_rotation"], 3, "ANNY global_rotation")[None],
        "global_translation": tensor1(params["global_translation"], 3, "ANNY global_translation")[None],
    }


def soma_params(model: SOMA, fixture: dict[str, Any]) -> dict[str, torch.Tensor]:
    params = fixture["params"]
    return {
        "body_pose": tensor2(params["body_pose"], 23, 3, "SOMA body_pose")[None],
        "head_pose": tensor2(params["head_pose"], 5, 3, "SOMA head_pose")[None],
        "hand_pose": tensor2(params["hand_pose"], 48, 3, "SOMA hand_pose")[None],
        "global_rotation": tensor1(params["global_rotation"], 3, "SOMA global_rotation")[None],
        "global_translation": tensor1(params["global_translation"], 3, "SOMA global_translation")[None],
    }


def garment_params(model: GarmentMeasurements, fixture: dict[str, Any]) -> dict[str, torch.Tensor]:
    params = fixture["params"]
    return {
        "shape": tensor1(params["shape"], model.num_shape_components, "GarmentMeasurements shape")[None],
        "body_pose": tensor2(params["body_pose"], 25, 3, "GarmentMeasurements body_pose")[None],
        "head_pose": tensor2(params["head_pose"], 3, 3, "GarmentMeasurements head_pose")[None],
        "hand_pose": tensor2(params["hand_pose"], 30, 3, "GarmentMeasurements hand_pose")[None],
        "pelvis_rotation": tensor1(params["pelvis_rotation"], 3, "GarmentMeasurements pelvis_rotation")[None],
        "global_rotation": tensor1(params["global_rotation"], 3, "GarmentMeasurements global_rotation")[None],
        "global_translation": tensor1(params["global_translation"], 3, "GarmentMeasurements global_translation")[None],
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


def export_anny(root: Path, out: Path) -> None:
    model = ANNY().eval()
    weights = model.weights
    write_json(
        out / "model_data" / "anny.json",
        {
            "template_vertices": weights.template_vertices,
            "blendshapes": weights.blendshapes,
            "template_bone_heads": weights.template_bone_heads,
            "template_bone_tails": weights.template_bone_tails,
            "bone_heads_blendshapes": weights.bone_heads_blendshapes,
            "bone_tails_blendshapes": weights.bone_tails_blendshapes,
            "bone_rolls_rotmat": weights.bone_rolls_rotmat,
            "phenotype_mask": weights.phenotype_mask,
            "lbs_joint_indices": weights.lbs_joint_indices,
            "lbs_joint_weights": weights.lbs_joint_weights,
            "faces": weights.faces,
            "parents": weights.parents,
        },
    )
    for fixture_path in sorted((root / "fixtures" / "anny").glob("*.json")):
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        params = anny_params(model, fixture)
        write_json(
            out / "reference" / "anny" / fixture_path.name,
            {
                "model": "anny",
                "case": fixture["case"],
                "skeleton": model.forward_skeleton(**params)[0],
                "mesh": model.forward_vertices(**params)[0],
            },
        )


def export_soma(root: Path, out: Path) -> None:
    model = SOMA().eval()
    weights = model.weights
    rest = model.get_rest_pose(dtype=weights.mean_active.dtype, hands="flat")
    pose = pack_soma_pose(
        torch,
        rest["global_rotation"][None],
        rest["body_pose"][None],
        rest["head_pose"][None],
        rest["hand_pose"][None],
    )
    prepared = model.prepare_identity(identity=rest["identity"][None], scale_params=None, pose=pose, cache=False)
    write_json(
        out / "model_data" / "soma.json",
        {
            "bind_shape_active": prepared.bind_shape_active[0],
            "world_bind_pose": prepared.world_bind_pose[0],
            "inverse_world_bind_pose": prepared.inverse_world_bind_pose[0],
            "t_pose_world": weights.t_pose_world,
            "corrective_bindpose": weights.correctives.corrective_bindpose,
            "corrective_W1": weights.correctives.corrective_W1,
            "corrective_W2_rows": weights.correctives.corrective_W2_rows,
            "corrective_W2_cols": weights.correctives.corrective_W2_cols,
            "corrective_W2_values": weights.correctives.corrective_W2_values,
            "skin_joint_indices": weights.skin_joint_indices_active,
            "skin_joint_weights": weights.skin_joint_weights_active,
            "faces": weights.faces,
            "parents": weights.topology.parents_full,
        },
    )
    for fixture_path in sorted((root / "fixtures" / "soma").glob("*.json")):
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        params = soma_params(model, fixture)
        write_json(
            out / "reference" / "soma" / fixture_path.name,
            {
                "model": "soma",
                "case": fixture["case"],
                "skeleton": model.forward_skeleton(**params)[0],
                "mesh": model.forward_vertices(**params)[0],
            },
        )


def export_garment(root: Path, out: Path) -> None:
    model = GarmentMeasurements().eval()
    weights = model.weights
    write_json(
        out / "model_data" / "garment.json",
        {
            "mean_vertices": weights.mean_vertices,
            "components": weights.components,
            "eigenvalues": weights.eigenvalues,
            "bind_quats": weights.bind_quats,
            "skin_joint_indices": weights.skin_joint_indices,
            "skin_joint_weights": weights.skin_joint_weights,
            "mvc_weights": weights.mvc_weights,
            "faces": weights.faces,
            "parents": weights.parents,
        },
    )
    for fixture_path in sorted((root / "fixtures" / "garment").glob("*.json")):
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        params = garment_params(model, fixture)
        write_json(
            out / "reference" / "garment" / fixture_path.name,
            {
                "model": "garment",
                "case": fixture["case"],
                "skeleton": model.forward_skeleton(**params)[0],
                "mesh": model.forward_vertices(**params)[0],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("generated"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("smpl", "mhr", "anny", "soma", "garment"),
        default=("smpl", "mhr", "anny", "soma", "garment"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = args.output if args.output.is_absolute() else root / args.output
    if "smpl" in args.models:
        export_smpl(root, out)
    if "mhr" in args.models:
        export_mhr(root, out)
    if "anny" in args.models:
        export_anny(root, out)
    if "soma" in args.models:
        export_soma(root, out)
    if "garment" in args.models:
        export_garment(root, out)


if __name__ == "__main__":
    main()
