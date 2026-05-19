import json
from pathlib import Path

import numpy as np
import torch

from body_models.anny.torch import ANNY
from body_models.mhr.numpy import MHR
from body_models.smpl.numpy import SMPL
from body_models.soma.pose import pack_pose as pack_soma_pose
from body_models.soma.torch import SOMA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "generated"


def dump_model(name, data):
    path = OUT / "model_data" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))


def dump_reference(name, case, skeleton, mesh):
    path = OUT / "reference" / name / f"{case}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model": name,
        "case": case,
        "skeleton": [transform.T.reshape(-1).tolist() for transform in skeleton],
        "mesh": mesh.tolist(),
    }, separators=(",", ":")))


def tensor_params(params):
    out = {}
    for name, value in params.items():
        array = np.asarray(value, dtype=np.float32)
        if array.ndim > 0:
            array = array[None]
        out[name] = torch.as_tensor(array, dtype=torch.float32)
    return out


def torch_json(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def mat3_json(mats):
    mats = torch_json(mats)
    return [mat.T.reshape(-1).tolist() for mat in np.asarray(mats)]


def mat4_json(mats):
    mats = torch_json(mats)
    return [mat.T.reshape(-1).tolist() for mat in np.asarray(mats)]


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
    for fixture_path in sorted((ROOT / "fixtures" / "smpl").glob("*.json")):
        fixture = json.loads(fixture_path.read_text())
        params = {name: np.asarray(value, dtype=np.float32)[None] for name, value in fixture["params"].items()}
        dump_reference("smpl", fixture["case"], smpl.forward_skeleton(**params)[0], smpl.forward_vertices(**params)[0])

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
    for fixture_path in sorted((ROOT / "fixtures" / "mhr").glob("*.json")):
        fixture = json.loads(fixture_path.read_text())
        params = {name: np.asarray(value, dtype=np.float32)[None] for name, value in fixture["params"].items()}
        dump_reference("mhr", fixture["case"], mhr.forward_skeleton(**params)[0], mhr.forward_vertices(**params)[0])

    anny = ANNY().eval()
    anny_weights = anny.weights
    dump_model("anny", {
        "template_vertices": anny_weights.template_vertices.detach().cpu().tolist(),
        "blendshapes": anny_weights.blendshapes.detach().cpu().tolist(),
        "template_bone_heads": anny_weights.template_bone_heads.detach().cpu().tolist(),
        "template_bone_tails": anny_weights.template_bone_tails.detach().cpu().tolist(),
        "bone_heads_blendshapes": anny_weights.bone_heads_blendshapes.detach().cpu().tolist(),
        "bone_tails_blendshapes": anny_weights.bone_tails_blendshapes.detach().cpu().tolist(),
        "bone_rolls_rotmat": mat3_json(anny_weights.bone_rolls_rotmat),
        "phenotype_mask": anny_weights.phenotype_mask.detach().cpu().tolist(),
        "lbs_joint_indices": anny_weights.lbs_joint_indices.detach().cpu().tolist(),
        "lbs_joint_weights": anny_weights.lbs_joint_weights.detach().cpu().tolist(),
        "faces": anny_weights.faces.detach().cpu().tolist(),
        "parents": anny_weights.parents,
    })
    for fixture_path in sorted((ROOT / "fixtures" / "anny").glob("*.json")):
        fixture = json.loads(fixture_path.read_text())
        params = tensor_params(fixture["params"])
        dump_reference(
            "anny",
            fixture["case"],
            anny.forward_skeleton(**params)[0].detach().cpu().numpy(),
            anny.forward_vertices(**params)[0].detach().cpu().numpy(),
        )

    soma = SOMA().eval()
    soma_weights = soma.weights
    rest = soma.get_rest_pose(dtype=soma_weights.mean_active.dtype, hands="flat")
    pose = pack_soma_pose(
        torch,
        rest["global_rotation"][None],
        rest["body_pose"][None],
        rest["head_pose"][None],
        rest["hand_pose"][None],
    )
    prepared = soma.prepare_identity(identity=rest["identity"][None], scale_params=None, pose=pose, cache=False)
    dump_model("soma", {
        "bind_shape_active": prepared.bind_shape_active[0].detach().cpu().tolist(),
        "world_bind_pose": mat4_json(prepared.world_bind_pose[0]),
        "inverse_world_bind_pose": mat4_json(prepared.inverse_world_bind_pose[0]),
        "t_pose_world": mat4_json(soma_weights.t_pose_world),
        "corrective_bindpose": mat3_json(soma_weights.correctives.corrective_bindpose),
        "corrective_W1": soma_weights.correctives.corrective_W1.detach().cpu().tolist(),
        "corrective_W2_rows": soma_weights.correctives.corrective_W2_rows.detach().cpu().tolist(),
        "corrective_W2_cols": soma_weights.correctives.corrective_W2_cols.detach().cpu().tolist(),
        "corrective_W2_values": soma_weights.correctives.corrective_W2_values.detach().cpu().tolist(),
        "skin_joint_indices": soma_weights.skin_joint_indices_active.detach().cpu().tolist(),
        "skin_joint_weights": soma_weights.skin_joint_weights_active.detach().cpu().tolist(),
        "faces": soma_weights.faces.detach().cpu().tolist(),
        "parents": soma_weights.topology.parents_full,
    })
    for fixture_path in sorted((ROOT / "fixtures" / "soma").glob("*.json")):
        fixture = json.loads(fixture_path.read_text())
        params = tensor_params(fixture["params"])
        dump_reference(
            "soma",
            fixture["case"],
            soma.forward_skeleton(**params)[0].detach().cpu().numpy(),
            soma.forward_vertices(**params)[0].detach().cpu().numpy(),
        )
