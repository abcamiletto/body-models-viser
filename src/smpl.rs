use anyhow::Result;

use crate::math;
use crate::types::{Mat3, Mat4, SmplModel, SmplParams, Vec3};

pub fn smpl_forward(model: &SmplModel, params: &SmplParams) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    math::ensure_len(&params.shape, 10, "SMPL shape")?;
    math::ensure_len(&params.body_pose, 23, "SMPL body_pose")?;

    let pose: Vec<Mat3> = std::iter::once(math::axis_angle_to_mat3(params.pelvis_rotation))
        .chain(
            params
                .body_pose
                .iter()
                .copied()
                .map(math::axis_angle_to_mat3),
        )
        .collect();
    let joints = shaped_joints(model, &params.shape);
    let skeleton = math::fk(
        &pose,
        &math::local_offsets(&joints, &model.parents),
        &model.parents,
    );
    let mut mesh = posed_vertices(model, params, &pose);
    skin_vertices(model, &joints, &skeleton, &mut mesh);

    math::apply_global_points(&mut mesh, params.global_rotation, params.global_translation);
    let skeleton =
        math::apply_global_skeleton(&skeleton, params.global_rotation, params.global_translation);
    Ok((skeleton, mesh))
}

fn shaped_joints(model: &SmplModel, shape: &[f64]) -> Vec<Vec3> {
    let mut joints = model.j_template.clone();
    for (j, joint) in joints.iter_mut().enumerate() {
        for (d, value) in joint.iter_mut().enumerate() {
            *value += math::dot(&model.j_shapedirs[j][d], shape);
        }
    }
    joints
}

fn posed_vertices(model: &SmplModel, params: &SmplParams, pose: &[Mat3]) -> Vec<Vec3> {
    let mut vertices = model.v_template.clone();
    for (v, vertex) in vertices.iter_mut().enumerate() {
        for (d, value) in vertex.iter_mut().enumerate() {
            *value += math::dot(&model.shapedirs[v][d], &params.shape);
        }
    }

    let pose_delta = pose_delta(pose);
    for (delta, row) in pose_delta.iter().zip(&model.posedirs) {
        if *delta == 0.0 {
            continue;
        }
        for (vertex, pose_delta) in vertices.iter_mut().zip(row.chunks_exact(3)) {
            vertex[0] += delta * pose_delta[0];
            vertex[1] += delta * pose_delta[1];
            vertex[2] += delta * pose_delta[2];
        }
    }
    vertices
}

fn pose_delta(pose: &[Mat3]) -> Vec<f64> {
    let mut delta = vec![0.0; 23 * 9];
    let eye = math::eye3();
    for j in 1..24 {
        for r in 0..3 {
            for c in 0..3 {
                delta[(j - 1) * 9 + r * 3 + c] = pose[j][r][c] - eye[r][c];
            }
        }
    }
    delta
}

fn skin_vertices(model: &SmplModel, joints: &[Vec3], skeleton: &[Mat4], vertices: &mut [Vec3]) {
    let joint_transforms: Vec<(Mat3, Vec3)> = (0..24)
        .map(|j| {
            let r = math::mat4_rot(&skeleton[j]);
            let t = math::mat4_trans(&skeleton[j]);
            (r, math::sub3(t, math::mat3_vec(r, joints[j])))
        })
        .collect();
    let sparse_weights = math::sparse_rows(&model.lbs_weights_sparse, &model.lbs_weights);

    for (v, vertex) in vertices.iter_mut().enumerate() {
        let mut wr = [[0.0; 3]; 3];
        let mut wt = [0.0; 3];
        for (j, w) in sparse_weights.row(v) {
            let (r, offset) = joint_transforms[j];
            for a in 0..3 {
                wt[a] += w * offset[a];
                for b in 0..3 {
                    wr[a][b] += w * r[a][b];
                }
            }
        }
        *vertex = math::add3(math::mat3_vec(wr, *vertex), wt);
    }
}
