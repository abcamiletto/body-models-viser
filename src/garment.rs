use anyhow::Result;

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
        .map(|(&quat, translation)| math::rt_to_mat4(wxyz_to_mat3(quat), translation))
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
            let pose_quat = axis_angle_to_quat_wxyz(pose);
            let quat = mul_wxyz(bind_quat, pose_quat);
            math::rt_to_mat4(wxyz_to_mat3(quat), translation)
        })
        .collect();
    fk(&posed_local, &model.parents)
}

fn local_translations(
    model: &GarmentModel,
    positions: &[Vec3],
    bind_global_quats: &[[f64; 4]],
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
                rotate_wxyz(inverse_wxyz(bind_global_quats[parent as usize]), offset)
            }
        })
        .collect()
}

fn propagate_quats(local: &[[f64; 4]], parents: &[isize]) -> Vec<[f64; 4]> {
    let mut global = vec![[0.0; 4]; local.len()];
    for joint in 0..local.len() {
        global[joint] = if parents[joint] < 0 {
            local[joint]
        } else {
            mul_wxyz(global[parents[joint] as usize], local[joint])
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

fn wxyz_to_mat3(q: [f64; 4]) -> crate::types::Mat3 {
    math::quat_xyzw_to_mat3([q[1], q[2], q[3], q[0]])
}

fn mul_wxyz(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let a = canonicalize_wxyz(a);
    let b = canonicalize_wxyz(b);
    canonicalize_wxyz([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])
}

fn inverse_wxyz(q: [f64; 4]) -> [f64; 4] {
    let [w, x, y, z] = canonicalize_wxyz(q);
    [w, -x, -y, -z]
}

fn rotate_wxyz(q: [f64; 4], point: Vec3) -> Vec3 {
    let [w, x, y, z] = canonicalize_wxyz(q);
    let [px, py, pz] = point;
    let cross1_x = y * pz - z * py + w * px;
    let cross1_y = z * px - x * pz + w * py;
    let cross1_z = x * py - y * px + w * pz;
    let cross2_x = y * cross1_z - z * cross1_y;
    let cross2_y = z * cross1_x - x * cross1_z;
    let cross2_z = x * cross1_y - y * cross1_x;
    [
        px + 2.0 * cross2_x,
        py + 2.0 * cross2_y,
        pz + 2.0 * cross2_z,
    ]
}

fn canonicalize_wxyz(q: [f64; 4]) -> [f64; 4] {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let sign = if q[0] < 0.0 { -1.0 } else { 1.0 };
    [
        sign * q[0] / norm,
        sign * q[1] / norm,
        sign * q[2] / norm,
        sign * q[3] / norm,
    ]
}

fn axis_angle_to_quat_wxyz(v: Vec3) -> [f64; 4] {
    let theta = math::norm3(v);
    if theta < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let scale = (theta * 0.5).sin() / theta;
    [
        (theta * 0.5).cos(),
        v[0] * scale,
        v[1] * scale,
        v[2] * scale,
    ]
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
