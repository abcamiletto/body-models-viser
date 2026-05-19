use anyhow::Result;

use crate::math;
use crate::types::{
    Mat3, Mat4, SmplFamilyModel, SmplModel, SmplParams, SmplhModel, SmplhParams, SmplxModel,
    SmplxParams, Vec3,
};

pub fn smpl_forward(model: &SmplModel, params: &SmplParams) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    math::ensure_len(&params.shape, 10, "SMPL shape")?;
    math::ensure_len(&params.body_pose, 23, "SMPL body_pose")?;

    let pose = pose_matrices(params.pelvis_rotation, [&params.body_pose]);
    forward(
        model,
        &params.shape,
        &[],
        &pose,
        params.global_rotation,
        params.global_translation,
    )
}

pub fn smplh_forward(model: &SmplhModel, params: &SmplhParams) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    math::ensure_len(&params.shape, 10, "SMPLH shape")?;
    math::ensure_len(&params.body_pose, 21, "SMPLH body_pose")?;
    math::ensure_len(&params.hand_pose, 30, "SMPLH hand_pose")?;

    let hand_pose = add_hand_mean(model, &params.hand_pose);
    let pose = pose_matrices(params.pelvis_rotation, [&params.body_pose, &hand_pose]);
    forward(
        model,
        &params.shape,
        &[],
        &pose,
        params.global_rotation,
        params.global_translation,
    )
}

pub fn smplx_forward(model: &SmplxModel, params: &SmplxParams) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    math::ensure_len(&params.shape, 10, "SMPLX shape")?;
    math::ensure_len(&params.expression, 10, "SMPLX expression")?;
    math::ensure_len(&params.body_pose, 21, "SMPLX body_pose")?;
    math::ensure_len(&params.head_pose, 3, "SMPLX head_pose")?;
    math::ensure_len(&params.hand_pose, 30, "SMPLX hand_pose")?;

    let hand_pose = add_hand_mean(model, &params.hand_pose);
    let pose = pose_matrices(
        params.pelvis_rotation,
        [&params.body_pose, &params.head_pose, &hand_pose],
    );
    forward(
        model,
        &params.shape,
        &params.expression,
        &pose,
        params.global_rotation,
        params.global_translation,
    )
}

fn forward(
    model: &SmplFamilyModel,
    shape: &[f64],
    expression: &[f64],
    pose: &[Mat3],
    global_rotation: Vec3,
    global_translation: Vec3,
) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    let joints = shaped_joints(model, shape, expression);
    let skeleton = math::fk(
        pose,
        &math::local_offsets(&joints, &model.parents),
        &model.parents,
    );
    let mut mesh = posed_vertices(model, shape, expression, pose);
    skin_vertices(model, &joints, &skeleton, &mut mesh);

    math::apply_global_points(&mut mesh, global_rotation, global_translation);
    let skeleton = math::apply_global_skeleton(&skeleton, global_rotation, global_translation);
    Ok((skeleton, mesh))
}

fn shaped_joints(model: &SmplFamilyModel, shape: &[f64], expression: &[f64]) -> Vec<Vec3> {
    let mut joints = model.j_template.clone();
    for (j, joint) in joints.iter_mut().enumerate() {
        for (d, value) in joint.iter_mut().enumerate() {
            *value += dot(&model.j_shapedirs[j][d], shape);
            *value += dot(
                model.j_exprdirs.get(j).map_or(&[][..], |dirs| &dirs[d]),
                expression,
            );
        }
    }
    joints
}

fn posed_vertices(
    model: &SmplFamilyModel,
    shape: &[f64],
    expression: &[f64],
    pose: &[Mat3],
) -> Vec<Vec3> {
    let mut vertices = model.v_template.clone();
    for (v, vertex) in vertices.iter_mut().enumerate() {
        for (d, value) in vertex.iter_mut().enumerate() {
            *value += dot(&model.shapedirs[v][d], shape);
            *value += dot(
                model.exprdirs.get(v).map_or(&[][..], |dirs| &dirs[d]),
                expression,
            );
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
    let mut delta = vec![0.0; (pose.len() - 1) * 9];
    let eye = math::eye3();
    for j in 1..pose.len() {
        for r in 0..3 {
            for c in 0..3 {
                delta[(j - 1) * 9 + r * 3 + c] = pose[j][r][c] - eye[r][c];
            }
        }
    }
    delta
}

fn skin_vertices(
    model: &SmplFamilyModel,
    joints: &[Vec3],
    skeleton: &[Mat4],
    vertices: &mut [Vec3],
) {
    let joint_transforms: Vec<(Mat3, Vec3)> = (0..model.parents.len())
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

fn pose_matrices<const N: usize>(pelvis: Vec3, parts: [&[Vec3]; N]) -> Vec<Mat3> {
    let joints = parts.iter().map(|part| part.len()).sum::<usize>() + 1;
    let mut pose = Vec::with_capacity(joints);
    pose.push(math::axis_angle_to_mat3(pelvis));
    for part in parts {
        pose.extend(part.iter().copied().map(math::axis_angle_to_mat3));
    }
    pose
}

fn add_hand_mean(model: &SmplFamilyModel, hand_pose: &[Vec3]) -> Vec<Vec3> {
    hand_pose
        .iter()
        .enumerate()
        .map(|(joint, &pose)| {
            let hand = joint / 15;
            let offset = joint % 15 * 3;
            [
                pose[0] + model.hand_mean[hand][offset],
                pose[1] + model.hand_mean[hand][offset + 1],
                pose[2] + model.hand_mean[hand][offset + 2],
            ]
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
