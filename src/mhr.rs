use anyhow::Result;
use std::f64::consts::LN_2;

use crate::math;
use crate::types::{Mat3, Mat4, MhrModel, MhrParams, Vec3};

pub fn mhr_forward(model: &MhrModel, params: &MhrParams) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    math::ensure_len(&params.shape, 45, "MHR shape")?;
    math::ensure_len(&params.body_pose, 100, "MHR body_pose")?;
    math::ensure_len(&params.hand_pose, 104, "MHR hand_pose")?;
    math::ensure_len(&params.expression, 72, "MHR expression")?;

    let pose = pack_pose(&params.body_pose, &params.hand_pose);
    let kinematics = skeleton_core(model, &pose);
    let skeleton = skeleton(
        &kinematics,
        params.global_rotation,
        params.global_translation,
    );
    let mut mesh = posed_vertices(model, params, &kinematics.joint_params);
    skin_vertices(model, &kinematics, &mut mesh);
    math::apply_global_points(&mut mesh, params.global_rotation, params.global_translation);

    Ok((skeleton, mesh))
}

struct Kinematics {
    translation: Vec<Vec3>,
    rotation: Vec<Mat3>,
    scale: Vec<f64>,
    joint_params: Vec<Vec<f64>>,
}

fn skeleton(kinematics: &Kinematics, rotation: Vec3, translation: Vec3) -> Vec<Mat4> {
    let local: Vec<Mat4> = (0..kinematics.translation.len())
        .map(|j| {
            math::trs_to_mat4(
                math::scale3(kinematics.translation[j], 0.01),
                kinematics.rotation[j],
                kinematics.scale[j],
            )
        })
        .collect();
    math::apply_global_skeleton(&local, rotation, translation)
}

fn posed_vertices(model: &MhrModel, params: &MhrParams, joint_params: &[Vec<f64>]) -> Vec<Vec3> {
    let coeffs: Vec<f64> = params
        .shape
        .iter()
        .chain(params.expression.iter())
        .copied()
        .collect();
    let mut vertices = model.base_vertices.clone();
    let blendshape_dirs =
        math::sparse_vec3_rows(&model.blendshape_dirs_sparse, &model.blendshape_dirs);
    for (i, coeff) in coeffs.iter().enumerate() {
        for (v, delta) in blendshape_dirs.row(i) {
            vertices[v][0] += coeff * delta[0];
            vertices[v][1] += coeff * delta[1];
            vertices[v][2] += coeff * delta[2];
        }
    }
    add_correctives(model, joint_params, &mut vertices);
    vertices
}

fn skin_vertices(model: &MhrModel, kinematics: &Kinematics, vertices: &mut [Vec3]) {
    let joint_transforms: Vec<(Mat3, Vec3)> = (0..kinematics.translation.len())
        .map(|joint| {
            let lin_g = math::scale_mat3(kinematics.rotation[joint], kinematics.scale[joint]);
            let lin = math::mat3_mul(lin_g, model.bind_inv_linear[joint]);
            let t = math::add3(
                math::mat3_vec(lin_g, model.bind_inv_translation[joint]),
                kinematics.translation[joint],
            );
            (lin, t)
        })
        .collect();

    for (v, vertex) in vertices.iter_mut().enumerate() {
        let mut out = [0.0; 3];
        for (k, &joint) in model.skin_indices[v].iter().enumerate() {
            let weight = model.skin_weights[v][k];
            if weight == 0.0 {
                continue;
            }
            let (lin, t) = joint_transforms[joint];
            out = math::add3(
                out,
                math::scale3(math::add3(math::mat3_vec(lin, *vertex), t), weight),
            );
        }
        *vertex = math::scale3(out, 0.01);
    }
}

fn add_correctives(model: &MhrModel, joint_params: &[Vec<f64>], vertices: &mut [Vec3]) {
    let mut feat = Vec::with_capacity((joint_params.len() - 2) * 6);
    for params in joint_params.iter().skip(2) {
        let rot = math::euler_xyz_to_mat3([params[3], params[4], params[5]]);
        feat.extend_from_slice(&[
            rot[0][0] - 1.0,
            rot[1][0],
            rot[2][0],
            rot[0][1],
            rot[1][1] - 1.0,
            rot[2][1],
        ]);
    }

    let corrective_w1 = math::sparse_rows(&model.corrective_w1_sparse, &model.corrective_w1);
    let hidden: Vec<f64> = (0..corrective_w1.len())
        .map(|row| math::sparse_dot(corrective_w1, row, &feat).max(0.0))
        .collect();
    let corrective_w2 = math::sparse_rows(&model.corrective_w2_sparse, &model.corrective_w2);
    for (v, vertex) in vertices.iter_mut().enumerate() {
        for (d, value) in vertex.iter_mut().enumerate() {
            *value += math::sparse_dot(corrective_w2, v * 3 + d, &hidden);
        }
    }
}

fn skeleton_core(model: &MhrModel, pose: &[f64]) -> Kinematics {
    let joints = model.parents.len();
    let mut joint_params = vec![vec![0.0; 7]; joints];
    let parameter_transform = math::sparse_rows(
        &model.parameter_transform_sparse,
        &model.parameter_transform,
    );
    for d in 0..parameter_transform.len() {
        joint_params[d / 7][d % 7] = math::sparse_dot(parameter_transform, d, pose);
    }

    let mut local_t = vec![[0.0; 3]; joints];
    let mut local_r = vec![math::eye3(); joints];
    let mut local_s = vec![1.0; joints];
    for j in 0..joints {
        local_t[j] = math::add3(
            [joint_params[j][0], joint_params[j][1], joint_params[j][2]],
            model.joint_offsets[j],
        );
        let q_local =
            math::euler_xyz_to_quat([joint_params[j][3], joint_params[j][4], joint_params[j][5]]);
        local_r[j] =
            math::quat_xyzw_to_mat3(math::quat_mul_xyzw(model.joint_pre_rotations[j], q_local));
        local_s[j] = (LN_2 * joint_params[j][6]).exp();
    }

    let mut global_t = vec![[0.0; 3]; joints];
    let mut global_r = vec![math::eye3(); joints];
    let mut global_s = vec![1.0; joints];
    for j in 0..joints {
        let parent = model.parents[j];
        if parent < 0 {
            global_t[j] = local_t[j];
            global_r[j] = local_r[j];
            global_s[j] = local_s[j];
        } else {
            let p = parent as usize;
            global_r[j] = math::mat3_mul(global_r[p], local_r[j]);
            global_s[j] = global_s[p] * local_s[j];
            global_t[j] = math::add3(
                math::mat3_vec(math::scale_mat3(global_r[p], global_s[p]), local_t[j]),
                global_t[p],
            );
        }
    }
    Kinematics {
        translation: global_t,
        rotation: global_r,
        scale: global_s,
        joint_params,
    }
}

fn pack_pose(body: &[f64], hand: &[f64]) -> Vec<f64> {
    let mut pose = Vec::with_capacity(204);
    pose.extend_from_slice(&body[..68]);
    pose.extend_from_slice(&hand[..54]);
    pose.extend_from_slice(&body[68..]);
    pose.extend_from_slice(&hand[54..]);
    pose
}
