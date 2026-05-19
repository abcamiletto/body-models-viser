import json
from pathlib import Path

import numpy as np

from body_models.smpl.numpy import SMPL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated"


if __name__ == "__main__":
    model = SMPL(gender="neutral")
    weights = model.weights

    model_data_path = OUT / "model_data" / "smpl.json"
    model_data_path.parent.mkdir(parents=True, exist_ok=True)
    model_data_path.write_text(json.dumps({
        "v_template": weights.v_template.tolist(),
        "faces": weights.faces.tolist(),
        "lbs_weights": weights.lbs_weights.tolist(),
        "shapedirs": weights.shapedirs[:, :, :10].tolist(),
        "posedirs": weights.posedirs.tolist(),
        "j_template": weights.j_template.tolist(),
        "j_shapedirs": weights.j_shapedirs[:, :, :10].tolist(),
        "parents": weights.parents,
    }, separators=(",", ":")))

    for fixture_path in sorted((ROOT / "fixtures" / "smpl").glob("*.json")):
        fixture = json.loads(fixture_path.read_text())
        params = {name: np.asarray(value, dtype=np.float32)[None] for name, value in fixture["params"].items()}
        skeleton = model.forward_skeleton(**params)[0]
        mesh = model.forward_vertices(**params)[0]

        reference_path = OUT / "reference" / "smpl" / fixture_path.name
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(json.dumps({
            "model": "smpl",
            "case": fixture["case"],
            "skeleton": [transform.T.reshape(-1).tolist() for transform in skeleton],
            "mesh": mesh.tolist(),
        }, separators=(",", ":")))
