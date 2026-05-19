use anyhow::Result;
use glam::{Mat3, Mat4, Quat, Vec3};

use crate::axis_angle_rigid_transform;
use crate::types::{MhrModel, MhrParams};

const LN_2: f32 = std::f32::consts::LN_2;

pub fn mhr_forward(model: &MhrModel, params: &MhrParams) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    ensure_len(&params.shape, 45, "MHR shape")?;
    ensure_len(&params.body_pose, 100, "MHR body_pose")?;
    ensure_len(&params.hand_pose, 104, "MHR hand_pose")?;
    ensure_len(&params.expression, 72, "MHR expression")?;

    let pose = pack_pose(&params.body_pose, &params.hand_pose);
    let kinematics = skeleton_core(model, &pose);
    let global = axis_angle_rigid_transform(params.global_rotation, params.global_translation);
    let skeleton = kinematics
        .translation
        .iter()
        .zip(&kinematics.rotation)
        .zip(&kinematics.scale)
        .map(|((&translation, &rotation), &scale)| {
            let linear = rotation * scale;
            let translation = translation * 0.01;
            global
                * Mat4::from_cols(
                    linear.x_axis.extend(0.0),
                    linear.y_axis.extend(0.0),
                    linear.z_axis.extend(0.0),
                    translation.extend(1.0),
                )
        })
        .collect();

    let mut mesh = posed_vertices(model, params, &kinematics.joint_params);
    skin_vertices(model, &kinematics, &mut mesh);
    for vertex in &mut mesh {
        *vertex = global.transform_point3(*vertex);
    }

    Ok((skeleton, mesh))
}

struct Kinematics {
    translation: Vec<Vec3>,
    rotation: Vec<Mat3>,
    scale: Vec<f32>,
    joint_params: Vec<[f32; 7]>,
}

fn posed_vertices(model: &MhrModel, params: &MhrParams, joint_params: &[[f32; 7]]) -> Vec<Vec3> {
    let coeffs: Vec<f32> = params
        .shape
        .iter()
        .chain(&params.expression)
        .copied()
        .collect();
    let mut vertices = model.base_vertices.clone();
    for (coeff, dirs) in coeffs.iter().zip(&model.blendshape_dirs) {
        for (vertex, &delta) in vertices.iter_mut().zip(dirs) {
            *vertex += *coeff * delta;
        }
    }
    add_correctives(model, joint_params, &mut vertices);
    vertices
}

fn skin_vertices(model: &MhrModel, kinematics: &Kinematics, vertices: &mut [Vec3]) {
    let mut joint_transforms = Vec::with_capacity(model.parents.len());
    for joint in 0..model.parents.len() {
        let rotation_scale = kinematics.rotation[joint] * kinematics.scale[joint];
        let linear = rotation_scale * mat3_from_rows(&model.bind_inv_linear[joint]);
        let translation =
            kinematics.translation[joint] + rotation_scale * model.bind_inv_translation[joint];
        joint_transforms.push(Mat4::from_cols(
            linear.x_axis.extend(0.0),
            linear.y_axis.extend(0.0),
            linear.z_axis.extend(0.0),
            translation.extend(1.0),
        ));
    }

    for (vertex, (weights, indices)) in vertices
        .iter_mut()
        .zip(model.skin_weights.iter().zip(&model.skin_indices))
    {
        let mut out = Vec3::ZERO;
        for (&weight, &joint) in weights.iter().zip(indices) {
            out += weight * joint_transforms[joint].transform_point3(*vertex);
        }
        *vertex = out * 0.01;
    }
}

fn add_correctives(model: &MhrModel, joint_params: &[[f32; 7]], vertices: &mut [Vec3]) {
    let mut features = Vec::with_capacity((joint_params.len() - 2) * 6);
    for params in joint_params.iter().skip(2) {
        let rot = euler_xyz_mat3(params[3], params[4], params[5]);
        features.extend_from_slice(&[
            rot.x_axis.x - 1.0,
            rot.x_axis.y,
            rot.x_axis.z,
            rot.y_axis.x,
            rot.y_axis.y - 1.0,
            rot.y_axis.z,
        ]);
    }

    let hidden: Vec<f32> = model
        .corrective_w1
        .iter()
        .map(|row| dot(row, &features).max(0.0))
        .collect();
    for (vertex_index, vertex) in vertices.iter_mut().enumerate() {
        vertex.x += dot(&model.corrective_w2[vertex_index * 3], &hidden);
        vertex.y += dot(&model.corrective_w2[vertex_index * 3 + 1], &hidden);
        vertex.z += dot(&model.corrective_w2[vertex_index * 3 + 2], &hidden);
    }
}

fn skeleton_core(model: &MhrModel, pose: &[f32]) -> Kinematics {
    let mut joint_params = vec![[0.0; 7]; model.parents.len()];
    for (index, row) in model.parameter_transform.iter().enumerate() {
        joint_params[index / 7][index % 7] = dot(row, pose);
    }

    let mut local_translation = vec![Vec3::ZERO; model.parents.len()];
    let mut local_rotation = vec![Mat3::IDENTITY; model.parents.len()];
    let mut local_scale = vec![1.0; model.parents.len()];
    for joint in 0..model.parents.len() {
        let params = joint_params[joint];
        local_translation[joint] =
            Vec3::new(params[0], params[1], params[2]) + model.joint_offsets[joint];
        let local = euler_xyz_quat(params[3], params[4], params[5]);
        let pre = Quat::from_xyzw(
            model.joint_pre_rotations[joint][0],
            model.joint_pre_rotations[joint][1],
            model.joint_pre_rotations[joint][2],
            model.joint_pre_rotations[joint][3],
        );
        local_rotation[joint] = Mat3::from_quat(pre * local);
        local_scale[joint] = (LN_2 * params[6]).exp();
    }

    let mut translation = vec![Vec3::ZERO; model.parents.len()];
    let mut rotation = vec![Mat3::IDENTITY; model.parents.len()];
    let mut scale = vec![1.0; model.parents.len()];
    for joint in 0..model.parents.len() {
        let parent = model.parents[joint];
        if parent < 0 {
            translation[joint] = local_translation[joint];
            rotation[joint] = local_rotation[joint];
            scale[joint] = local_scale[joint];
        } else {
            let parent = parent as usize;
            rotation[joint] = rotation[parent] * local_rotation[joint];
            scale[joint] = scale[parent] * local_scale[joint];
            translation[joint] =
                rotation[parent] * scale[parent] * local_translation[joint] + translation[parent];
        }
    }
    Kinematics {
        translation,
        rotation,
        scale,
        joint_params,
    }
}

fn pack_pose(body: &[f32], hand: &[f32]) -> Vec<f32> {
    let mut pose = Vec::with_capacity(204);
    pose.extend_from_slice(&body[..68]);
    pose.extend_from_slice(&hand[..54]);
    pose.extend_from_slice(&body[68..]);
    pose.extend_from_slice(&hand[54..]);
    pose
}

fn mat3_from_rows(rows: &[[f32; 3]; 3]) -> Mat3 {
    Mat3::from_cols_array(&[
        rows[0][0], rows[1][0], rows[2][0], rows[0][1], rows[1][1], rows[2][1], rows[0][2],
        rows[1][2], rows[2][2],
    ])
}

fn euler_xyz_mat3(x: f32, y: f32, z: f32) -> Mat3 {
    let (sx, cx) = x.sin_cos();
    let (sy, cy) = y.sin_cos();
    let (sz, cz) = z.sin_cos();
    Mat3::from_cols_array(&[
        cz * cy,
        sz * cy,
        -sy,
        cz * sy * sx - sz * cx,
        sz * sy * sx + cz * cx,
        cy * sx,
        cz * sy * cx + sz * sx,
        sz * sy * cx - cz * sx,
        cy * cx,
    ])
}

fn euler_xyz_quat(x: f32, y: f32, z: f32) -> Quat {
    let (sx, cx) = (x * 0.5).sin_cos();
    let (sy, cy) = (y * 0.5).sin_cos();
    let (sz, cz) = (z * 0.5).sin_cos();
    Quat::from_xyzw(
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz,
    )
}

fn ensure_len<T>(values: &[T], len: usize, name: &str) -> Result<()> {
    anyhow::ensure!(
        values.len() == len,
        "expected {name} length {len}, got {}",
        values.len()
    );
    Ok(())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
