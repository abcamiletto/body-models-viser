use anyhow::Result;

use crate::math;
use crate::types::{Mat3, Mat4, SomaModel, SomaParams, Vec3};

pub fn soma_forward(model: &SomaModel, params: &SomaParams) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    math::ensure_len(&params.body_pose, 23, "SOMA body_pose")?;
    math::ensure_len(&params.head_pose, 5, "SOMA head_pose")?;
    math::ensure_len(&params.hand_pose, 48, "SOMA hand_pose")?;

    let pose = pack_pose(params);
    let pose_rot = oriented_pose_rotations(model, &pose);
    let mut skeleton_full = pose_skeleton(model, &pose_rot);
    let correctives = pose_correctives(model, &pose_rot);
    let rest_shape = corrected_shape(model, &correctives);
    let bone_transforms: Vec<Mat4> = skeleton_full
        .iter()
        .zip(&model.inverse_world_bind_pose)
        .map(|(&world, &inverse_bind)| math::mat4_mul(world, inverse_bind))
        .collect();
    let mut mesh = skin_vertices(model, &rest_shape, &bone_transforms);

    for vertex in &mut mesh {
        *vertex = math::scale3(*vertex, 0.01);
    }
    for transform in &mut skeleton_full {
        transform[0][3] *= 0.01;
        transform[1][3] *= 0.01;
        transform[2][3] *= 0.01;
    }

    math::apply_global_points(&mut mesh, [0.0; 3], params.global_translation);
    let skeleton_full =
        math::apply_global_skeleton(&skeleton_full, [0.0; 3], params.global_translation);
    Ok((skeleton_full[1..].to_vec(), mesh))
}

fn oriented_pose_rotations(model: &SomaModel, pose: &[Vec3]) -> Vec<Mat3> {
    let mut pose_rot = Vec::with_capacity(model.parents.len());
    pose_rot.push(math::eye3());
    pose_rot.extend(pose.iter().copied().map(math::axis_angle_to_mat3));

    pose_rot
        .iter()
        .enumerate()
        .map(|(joint, &rotation)| {
            let parent = model.parents[joint] as usize;
            let orient_parent = math::mat3_transpose(math::mat4_rot(&model.t_pose_world[parent]));
            let orient = math::mat4_rot(&model.t_pose_world[joint]);
            math::mat3_mul(math::mat3_mul(orient_parent, rotation), orient)
        })
        .collect()
}

fn pose_skeleton(model: &SomaModel, pose_rot: &[Mat3]) -> Vec<Mat4> {
    let bind_local = joint_world_to_local(&model.world_bind_pose, &model.parents);
    let mut local_translations: Vec<Vec3> = bind_local.iter().map(math::mat4_trans).collect();
    local_translations[1] = [0.0; 3];

    let local: Vec<Mat4> = pose_rot
        .iter()
        .copied()
        .zip(local_translations)
        .map(|(rotation, translation)| math::rt_to_mat4(rotation, translation))
        .collect();
    fk_full(&local, &model.parents)
}

fn pose_correctives(model: &SomaModel, pose_rot: &[Mat3]) -> Vec<Vec3> {
    let mut features = Vec::with_capacity(model.corrective_bindpose.len() * 6);
    for (&bindpose, &rotation) in model.corrective_bindpose.iter().zip(pose_rot) {
        let mut x = math::mat3_mul(math::mat3_transpose(bindpose), rotation);
        x[0][0] -= 1.0;
        x[1][1] -= 1.0;
        features.extend_from_slice(&[x[0][0], x[0][1], x[1][0], x[1][1], x[2][0], x[2][1]]);
    }

    let hidden: Vec<f64> = (0..model.corrective_w1[0].len())
        .map(|col| {
            model
                .corrective_w1
                .iter()
                .zip(&features)
                .map(|(row, feature)| row[col] * feature)
                .sum::<f64>()
                .max(0.0)
        })
        .collect();
    let mut offsets = vec![[0.0; 3]; model.bind_shape_active.len()];
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
        .map(|(&vertex, &offset)| math::add3(vertex, offset))
        .collect()
}

fn skin_vertices(model: &SomaModel, rest_shape: &[Vec3], bone_transforms: &[Mat4]) -> Vec<Vec3> {
    rest_shape
        .iter()
        .enumerate()
        .map(|(vertex, &rest)| {
            let mut out = [0.0; 3];
            for (slot, &joint) in model.skin_joint_indices[vertex].iter().enumerate() {
                if joint < 0 {
                    continue;
                }
                let weight = model.skin_joint_weights[vertex][slot];
                let joint = joint as usize;
                let transformed = math::add3(
                    math::mat3_vec(math::mat4_rot(&bone_transforms[joint]), rest),
                    math::mat4_trans(&bone_transforms[joint]),
                );
                out = math::add3(out, math::scale3(transformed, weight));
            }
            out
        })
        .collect()
}

fn joint_world_to_local(world: &[Mat4], parents: &[isize]) -> Vec<Mat4> {
    let inverse: Vec<Mat4> = world.iter().copied().map(math::invert_rigid).collect();
    world
        .iter()
        .enumerate()
        .map(|(joint, &transform)| math::mat4_mul(inverse[parents[joint] as usize], transform))
        .collect()
}

fn fk_full(local: &[Mat4], parents: &[isize]) -> Vec<Mat4> {
    let mut world = vec![[[0.0; 4]; 4]; local.len()];
    for joint in 0..local.len() {
        world[joint] = if parents[joint] == joint as isize || parents[joint] < 0 {
            local[joint]
        } else {
            math::mat4_mul(world[parents[joint] as usize], local[joint])
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
