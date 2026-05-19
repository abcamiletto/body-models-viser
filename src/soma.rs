use anyhow::Result;
use glam::{Mat3, Mat4, Quat, Vec3};

use crate::axis_angle_rigid_transform;
use crate::types::{SomaModel, SomaParams};

pub fn soma_forward(model: &SomaModel, params: &SomaParams) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    ensure_len(&params.body_pose, 23, "SOMA body_pose")?;
    ensure_len(&params.head_pose, 5, "SOMA head_pose")?;
    ensure_len(&params.hand_pose, 48, "SOMA hand_pose")?;

    let pose = pack_pose(params);
    let pose_rot = oriented_pose_rotations(model, &pose);
    let mut skeleton_full = pose_skeleton(model, &pose_rot);
    let correctives = pose_correctives(model, &pose_rot);
    let rest_shape = corrected_shape(model, &correctives);
    let bone_transforms: Vec<Mat4> = skeleton_full
        .iter()
        .zip(&model.inverse_world_bind_pose)
        .map(|(&world, &inverse_bind)| world * inverse_bind)
        .collect();
    let mut mesh = skin_vertices(model, &rest_shape, &bone_transforms);

    for vertex in &mut mesh {
        *vertex *= 0.01;
    }
    for transform in &mut skeleton_full {
        transform.w_axis.x *= 0.01;
        transform.w_axis.y *= 0.01;
        transform.w_axis.z *= 0.01;
    }

    let global = axis_angle_rigid_transform(Vec3::ZERO, params.global_translation);
    for vertex in &mut mesh {
        *vertex = global.transform_point3(*vertex);
    }
    let skeleton_full: Vec<Mat4> = skeleton_full
        .into_iter()
        .map(|transform| global * transform)
        .collect();
    Ok((skeleton_full[1..].to_vec(), mesh))
}

fn oriented_pose_rotations(model: &SomaModel, pose: &[Vec3]) -> Vec<Mat3> {
    let mut pose_rot = Vec::with_capacity(model.parents.len());
    pose_rot.push(Mat3::IDENTITY);
    pose_rot.extend(
        pose.iter()
            .map(|&rotation| Mat3::from_quat(axis_angle_quat(rotation))),
    );

    pose_rot
        .iter()
        .enumerate()
        .map(|(joint, &rotation)| {
            let parent = model.parents[joint] as usize;
            let orient_parent = Mat3::from_mat4(model.t_pose_world[parent]).transpose();
            let orient = Mat3::from_mat4(model.t_pose_world[joint]);
            orient_parent * rotation * orient
        })
        .collect()
}

fn pose_skeleton(model: &SomaModel, pose_rot: &[Mat3]) -> Vec<Mat4> {
    let bind_local = joint_world_to_local(&model.world_bind_pose, &model.parents);
    let mut local_translations: Vec<Vec3> = bind_local
        .iter()
        .map(|transform| transform.w_axis.truncate())
        .collect();
    local_translations[1] = Vec3::ZERO;

    let local: Vec<Mat4> = pose_rot
        .iter()
        .copied()
        .zip(local_translations)
        .map(|(rotation, translation)| mat4_from_mat3_translation(rotation, translation))
        .collect();
    fk_full(&local, &model.parents)
}

fn pose_correctives(model: &SomaModel, pose_rot: &[Mat3]) -> Vec<Vec3> {
    let mut features = Vec::with_capacity(model.corrective_bindpose.len() * 6);
    for (&bindpose, &rotation) in model.corrective_bindpose.iter().zip(pose_rot) {
        let mut x = bindpose.transpose() * rotation;
        x.x_axis.x -= 1.0;
        x.y_axis.y -= 1.0;
        features.extend_from_slice(&[
            x.x_axis.x, x.y_axis.x, x.x_axis.y, x.y_axis.y, x.x_axis.z, x.y_axis.z,
        ]);
    }

    let hidden: Vec<f32> = (0..model.corrective_w1[0].len())
        .map(|col| {
            model
                .corrective_w1
                .iter()
                .zip(&features)
                .map(|(row, feature)| row[col] * feature)
                .sum::<f32>()
                .max(0.0)
        })
        .collect();
    let mut offsets = vec![Vec3::ZERO; model.bind_shape_active.len()];
    for ((&row, &col), &value) in model
        .corrective_w2_rows
        .iter()
        .zip(&model.corrective_w2_cols)
        .zip(&model.corrective_w2_values)
    {
        offsets[col / 3][col % 3] += hidden[row] * value;
    }
    offsets
}

fn corrected_shape(model: &SomaModel, correctives: &[Vec3]) -> Vec<Vec3> {
    model
        .bind_shape_active
        .iter()
        .zip(correctives)
        .map(|(&vertex, &offset)| vertex + offset)
        .collect()
}

fn skin_vertices(model: &SomaModel, rest_shape: &[Vec3], bone_transforms: &[Mat4]) -> Vec<Vec3> {
    rest_shape
        .iter()
        .enumerate()
        .map(|(vertex, &rest)| {
            let mut out = Vec3::ZERO;
            for (slot, &joint) in model.skin_joint_indices[vertex].iter().enumerate() {
                if joint < 0 {
                    continue;
                }
                let weight = model.skin_joint_weights[vertex][slot];
                out += weight * bone_transforms[joint as usize].transform_point3(rest);
            }
            out
        })
        .collect()
}

fn joint_world_to_local(world: &[Mat4], parents: &[isize]) -> Vec<Mat4> {
    let inverse: Vec<Mat4> = world.iter().map(|transform| transform.inverse()).collect();
    world
        .iter()
        .enumerate()
        .map(|(joint, &transform)| inverse[parents[joint] as usize] * transform)
        .collect()
}

fn fk_full(local: &[Mat4], parents: &[isize]) -> Vec<Mat4> {
    let mut world = vec![Mat4::IDENTITY; local.len()];
    for joint in 0..local.len() {
        world[joint] = if parents[joint] == joint as isize || parents[joint] < 0 {
            local[joint]
        } else {
            world[parents[joint] as usize] * local[joint]
        };
    }
    world
}

fn pack_pose(params: &SomaParams) -> Vec<Vec3> {
    let mut pose = Vec::with_capacity(77);
    pose.push(params.global_rotation);
    pose.extend_from_slice(&params.body_pose[..5]);
    pose.extend_from_slice(&params.head_pose);
    pose.extend_from_slice(&params.body_pose[5..9]);
    pose.extend_from_slice(&params.hand_pose[..24]);
    pose.extend_from_slice(&params.body_pose[9..13]);
    pose.extend_from_slice(&params.hand_pose[24..]);
    pose.extend_from_slice(&params.body_pose[13..]);
    pose
}

fn mat4_from_mat3_translation(linear: Mat3, translation: Vec3) -> Mat4 {
    Mat4::from_cols(
        linear.x_axis.extend(0.0),
        linear.y_axis.extend(0.0),
        linear.z_axis.extend(0.0),
        translation.extend(1.0),
    )
}

fn axis_angle_quat(axis_angle: Vec3) -> Quat {
    Quat::from_axis_angle(axis_angle.normalize_or_zero(), axis_angle.length())
}

fn ensure_len<T>(values: &[T], len: usize, name: &str) -> Result<()> {
    anyhow::ensure!(
        values.len() == len,
        "expected {name} length {len}, got {}",
        values.len()
    );
    Ok(())
}
