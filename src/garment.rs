use anyhow::Result;
use glam::{DQuat, DVec3};

use crate::math;
use crate::types::{GarmentModel, GarmentParams, Mat4, Vec3};

pub fn garment_forward(
    model: &GarmentModel,
    params: &GarmentParams,
) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    math::ensure_len(&params.shape, 15, "GarmentMeasurements shape")?;
    math::ensure_len(&params.body_pose, 25, "GarmentMeasurements body_pose")?;
    math::ensure_len(&params.head_pose, 3, "GarmentMeasurements head_pose")?;
    math::ensure_len(&params.hand_pose, 30, "GarmentMeasurements hand_pose")?;

    let pose = pack_pose(params);
    let shaped_vertices = shaped_vertices(model, &params.shape);
    let joint_positions = joint_positions(model, &shaped_vertices);
    let bind_global = bind_skeleton(model, &joint_positions);
    let posed_global = posed_skeleton(model, &pose, &bind_global);
    let bone_transforms: Vec<Mat4> = posed_global
        .iter()
        .zip(&bind_global)
        .map(|(&posed, &bind)| math::mat4_mul(posed, math::invert_rigid(bind)))
        .collect();
    let mut mesh = skin_vertices(model, &shaped_vertices, &bone_transforms);
    math::apply_global_points(&mut mesh, params.global_rotation, params.global_translation);
    let skeleton = math::apply_global_skeleton(
        &posed_global,
        params.global_rotation,
        params.global_translation,
    );
    Ok((skeleton, mesh))
}

fn shaped_vertices(model: &GarmentModel, shape: &[f64]) -> Vec<Vec3> {
    let scaled_shape: Vec<f64> = shape
        .iter()
        .zip(&model.eigenvalues)
        .map(|(value, eigenvalue)| value * eigenvalue.sqrt())
        .collect();
    let mut vertices = model.mean_vertices.clone();
    for (vertex, components) in vertices.iter_mut().zip(&model.components) {
        for d in 0..3 {
            vertex[d] += components[d]
                .iter()
                .zip(&scaled_shape)
                .map(|(component, coeff)| component * coeff)
                .sum::<f64>();
        }
    }
    vertices
}

fn joint_positions(model: &GarmentModel, vertices: &[Vec3]) -> Vec<Vec3> {
    let mut joints = vec![[0.0; 3]; model.parents.len()];
    for (vertex, weights) in vertices.iter().zip(&model.mvc_weights) {
        for (joint, weight) in weights.iter().enumerate() {
            joints[joint][0] += weight * vertex[0];
            joints[joint][1] += weight * vertex[1];
            joints[joint][2] += weight * vertex[2];
        }
    }
    joints
}

fn bind_skeleton(model: &GarmentModel, joint_positions: &[Vec3]) -> Vec<Mat4> {
    let bind_global_quats = propagate_quats(&model.bind_quats, &model.parents);
    let translations = local_translations(model, joint_positions, &bind_global_quats);
    let bind_local: Vec<Mat4> = model
        .bind_quats
        .iter()
        .zip(translations)
        .map(|(&quat, translation)| math::rt_to_mat4(quat_to_mat3(wxyz_quat(quat)), translation))
        .collect();
    fk(&bind_local, &model.parents)
}

fn posed_skeleton(model: &GarmentModel, pose: &[Vec3], bind_global: &[Mat4]) -> Vec<Mat4> {
    let bind_translations: Vec<Vec3> = joint_world_to_local(bind_global, &model.parents)
        .iter()
        .map(math::mat4_trans)
        .collect();
    let posed_local: Vec<Mat4> = model
        .bind_quats
        .iter()
        .zip(pose)
        .zip(bind_translations)
        .map(|((&bind_quat, &pose), translation)| {
            let quat = wxyz_quat(bind_quat) * axis_angle_quat(pose);
            math::rt_to_mat4(quat_to_mat3(quat), translation)
        })
        .collect();
    fk(&posed_local, &model.parents)
}

fn local_translations(
    model: &GarmentModel,
    positions: &[Vec3],
    bind_global_quats: &[DQuat],
) -> Vec<Vec3> {
    positions
        .iter()
        .enumerate()
        .map(|(joint, &position)| {
            let parent = model.parents[joint];
            if parent < 0 {
                position
            } else {
                let offset = math::sub3(position, positions[parent as usize]);
                rotate(bind_global_quats[parent as usize].inverse(), offset)
            }
        })
        .collect()
}

fn propagate_quats(local: &[[f64; 4]], parents: &[isize]) -> Vec<DQuat> {
    let mut global = vec![DQuat::IDENTITY; local.len()];
    for joint in 0..local.len() {
        global[joint] = if parents[joint] < 0 {
            wxyz_quat(local[joint])
        } else {
            global[parents[joint] as usize] * wxyz_quat(local[joint])
        };
    }
    global
}

fn skin_vertices(model: &GarmentModel, vertices: &[Vec3], transforms: &[Mat4]) -> Vec<Vec3> {
    vertices
        .iter()
        .enumerate()
        .map(|(vertex, &rest)| {
            let mut out = [0.0; 3];
            for (slot, &joint) in model.skin_joint_indices[vertex].iter().enumerate() {
                let weight = model.skin_joint_weights[vertex][slot];
                let transformed = math::add3(
                    math::mat3_vec(math::mat4_rot(&transforms[joint]), rest),
                    math::mat4_trans(&transforms[joint]),
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
        .map(|(joint, &transform)| {
            if parents[joint] < 0 {
                transform
            } else {
                math::mat4_mul(inverse[parents[joint] as usize], transform)
            }
        })
        .collect()
}

fn fk(local: &[Mat4], parents: &[isize]) -> Vec<Mat4> {
    let mut world = vec![[[0.0; 4]; 4]; local.len()];
    for joint in 0..local.len() {
        world[joint] = if parents[joint] < 0 {
            local[joint]
        } else {
            math::mat4_mul(world[parents[joint] as usize], local[joint])
        };
    }
    world
}

fn wxyz_quat(q: [f64; 4]) -> DQuat {
    DQuat::from_xyzw(q[1], q[2], q[3], q[0]).normalize()
}

fn quat_to_mat3(q: DQuat) -> crate::types::Mat3 {
    math::quat_xyzw_to_mat3([q.x, q.y, q.z, q.w])
}

fn rotate(q: DQuat, point: Vec3) -> Vec3 {
    let point = q * DVec3::from_array(point);
    point.to_array()
}

fn axis_angle_quat(v: Vec3) -> DQuat {
    let theta = math::norm3(v);
    if theta < 1e-12 {
        return DQuat::IDENTITY;
    }
    DQuat::from_axis_angle(DVec3::from_array(v) / theta, theta)
}

fn pack_pose(params: &GarmentParams) -> Vec<Vec3> {
    let mut pose = Vec::with_capacity(59);
    pose.push(params.pelvis_rotation);
    pose.extend_from_slice(&params.body_pose[..5]);
    pose.extend_from_slice(&params.head_pose);
    pose.extend_from_slice(&params.body_pose[5..11]);
    pose.extend_from_slice(&params.hand_pose[..15]);
    pose.extend_from_slice(&params.body_pose[11..17]);
    pose.extend_from_slice(&params.hand_pose[15..]);
    pose.extend_from_slice(&params.body_pose[17..]);
    pose
}
