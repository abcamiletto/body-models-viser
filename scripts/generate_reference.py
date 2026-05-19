import json
from pathlib import Path

import numpy as np

from body_models.mhr.numpy import MHR
from body_models.smpl.numpy import SMPL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated"


def dump_model(name, data):
    path = OUT / "model_data" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))


def dump_references(body_model, name):
    for fixture_path in sorted((ROOT / "fixtures" / name).glob("*.json")):
        fixture = json.loads(fixture_path.read_text())
        params = {
            param_name: np.asarray(value, dtype=np.float32)[None]
            for param_name, value in fixture["params"].items()
        }
        skeleton = body_model.forward_skeleton(**params)[0]
        mesh = body_model.forward_vertices(**params)[0]
        path = OUT / "reference" / name / fixture_path.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model": name,
            "case": fixture["case"],
            "skeleton": [transform.T.reshape(-1).tolist() for transform in skeleton],
            "mesh": mesh.tolist(),
        }, separators=(",", ":")))


if __name__ == "__main__":
    smpl = SMPL(gender="neutral")
    smpl_weights = smpl.weights
    dump_model("smpl", {
        "v_template": smpl_weights.v_template.tolist(),
        "faces": smpl_weights.faces.tolist(),
        "lbs_weights": smpl_weights.lbs_weights.tolist(),
        "shapedirs": smpl_weights.shapedirs[:, :, :10].tolist(),
        "posedirs": smpl_weights.posedirs.tolist(),
        "j_template": smpl_weights.j_template.tolist(),
        "j_shapedirs": smpl_weights.j_shapedirs[:, :, :10].tolist(),
        "parents": smpl_weights.parents,
    })
    dump_references(smpl, "smpl")

    mhr = MHR()
    mhr_weights = mhr.weights
    dump_model("mhr", {
        "base_vertices": mhr_weights.base_vertices.tolist(),
        "blendshape_dirs": mhr_weights.blendshape_dirs.tolist(),
        "skin_weights": mhr_weights.skin_weights.tolist(),
        "skin_indices": mhr_weights.skin_indices.tolist(),
        "faces": mhr_weights.faces.tolist(),
        "joint_offsets": mhr_weights.joint_offsets.tolist(),
        "joint_pre_rotations": mhr_weights.joint_pre_rotations.tolist(),
        "parameter_transform": mhr_weights.parameter_transform.tolist(),
        "bind_inv_linear": mhr_weights.bind_inv_linear.tolist(),
        "bind_inv_translation": mhr_weights.bind_inv_translation.tolist(),
        "corrective_W1": mhr_weights.corrective_W1.tolist(),
        "corrective_W2": mhr_weights.corrective_W2.tolist(),
        "parents": mhr_weights.parents,
    })
    dump_references(mhr, "mhr")
