use anyhow::Result;

use crate::math;
use crate::types::{AnnyModel, AnnyParams, Mat3, Mat4, Vec3};

const PHENOTYPE_ANCHORS: [&[f64]; 8] = [
    &[0.0, 1.0],
    &[-1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
    &[0.0, 0.5, 1.0],
    &[0.0, 0.5, 1.0],
    &[0.0, 1.0],
    &[0.0, 1.0],
    &[0.0, 0.5, 1.0],
    &[0.0, 0.5, 1.0],
];

pub fn anny_forward(model: &AnnyModel, params: &AnnyParams) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    math::ensure_len(&params.body_pose, 64, "ANNY body_pose")?;
    math::ensure_len(&params.head_pose, 60, "ANNY head_pose")?;
    math::ensure_len(&params.hand_pose, 38, "ANNY hand_pose")?;

    let pose = pack_pose(params);
    let pose_transforms: Vec<Mat4> = pose
        .iter()
        .copied()
        .map(|rotation| math::rt_to_mat4(math::axis_angle_to_mat3(rotation), [0.0; 3]))
        .collect();
    let coeffs = phenotype_coeffs(model, params);
    let rest_poses = rest_poses(model, &coeffs);
    let (mut skeleton, bone_transforms) = forward_kinematics(model, &rest_poses, &pose_transforms);
    let mut mesh = skin_vertices(model, &coeffs, &bone_transforms);

    math::apply_global_points(&mut mesh, [0.0; 3], params.global_translation);
    skeleton = math::apply_global_skeleton(&skeleton, [0.0; 3], params.global_translation);
    Ok((skeleton, mesh))
}

fn phenotype_coeffs(model: &AnnyModel, params: &AnnyParams) -> Vec<f64> {
    let values = [
        params.gender,
        params.age,
        params.muscle,
        params.weight,
        params.height,
        params.proportions,
        0.5,
        0.5,
    ];
    let mut weights = Vec::with_capacity(26);
    weights.extend_from_slice(&[1.0 / 3.0; 3]);
    for (value, anchors) in values.iter().zip(PHENOTYPE_ANCHORS) {
        weights.extend(interpolation_weights(*value, anchors));
    }

    model
        .phenotype_mask
        .iter()
        .map(|mask| {
            mask.iter()
                .zip(&weights)
                .map(|(active, weight)| if *active == 0.0 { 1.0 } else { *weight })
                .product()
        })
        .collect()
}

fn interpolation_weights(value: f64, anchors: &[f64]) -> Vec<f64> {
    let upper = anchors
        .iter()
        .position(|anchor| value < *anchor)
        .unwrap_or(anchors.len() - 1)
        .max(1);
    let lower = upper - 1;
    let alpha = ((value - anchors[lower]) / (anchors[upper] - anchors[lower])).clamp(0.0, 1.0);
    let mut weights = vec![0.0; anchors.len()];
    weights[lower] = 1.0 - alpha;
    weights[upper] = alpha;
    weights
}

fn rest_poses(model: &AnnyModel, coeffs: &[f64]) -> Vec<Mat4> {
    let heads = blended_points(
        &model.template_bone_heads,
        &model.bone_heads_blendshapes,
        coeffs,
    );
    let tails = blended_points(
        &model.template_bone_tails,
        &model.bone_tails_blendshapes,
        coeffs,
    );
    heads
        .iter()
        .zip(tails)
        .zip(&model.bone_rolls_rotmat)
        .map(|((&head, tail), &roll)| math::rt_to_mat4(bone_rotation(head, tail, roll), head))
        .collect()
}

fn bone_rotation(head: Vec3, tail: Vec3, roll: Mat3) -> Mat3 {
    let y_axis = [0.0, 1.0, 0.0];
    let y = math::scale3(
        math::sub3(tail, head),
        1.0 / math::norm3(math::sub3(tail, head)),
    );
    let cross = math::cross3(y, y_axis);
    let cross_norm = math::norm3(cross);
    let dot = y[0] * y_axis[0] + y[1] * y_axis[1] + y[2] * y_axis[2];
    let axis = math::scale3(cross, 1.0 / cross_norm);
    let angle = cross_norm.atan2(dot);
    let rotation = if (math::norm3(axis).powi(2) - 1.0).abs() < 0.1 {
        math::axis_angle_to_mat3(math::scale3(axis, -angle))
    } else {
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    };
    math::mat3_mul(rotation, roll)
}

fn forward_kinematics(
    model: &AnnyModel,
    rest_poses: &[Mat4],
    pose: &[Mat4],
) -> (Vec<Mat4>, Vec<Mat4>) {
    let root_rest = rest_poses[0];
    let base = math::invert_rigid(root_rest);
    let root_rotation = math::rt_to_mat4(math::mat4_rot(&root_rest), [0.0; 3]);
    let mut delta = pose.to_vec();
    delta[0] = math::mat4_mul(pose[0], root_rotation);

    let rest_inv: Vec<Mat4> = rest_poses.iter().copied().map(math::invert_rigid).collect();
    let local: Vec<Mat4> = rest_poses
        .iter()
        .zip(delta)
        .map(|(&rest, delta)| math::mat4_mul(rest, delta))
        .collect();
    let mut skeleton = vec![[[0.0; 4]; 4]; rest_poses.len()];
    let mut transforms = vec![[[0.0; 4]; 4]; rest_poses.len()];
    for joint in 0..rest_poses.len() {
        skeleton[joint] = if model.parents[joint] < 0 {
            math::mat4_mul(base, local[joint])
        } else {
            math::mat4_mul(transforms[model.parents[joint] as usize], local[joint])
        };
        transforms[joint] = math::mat4_mul(skeleton[joint], rest_inv[joint]);
    }
    (skeleton, transforms)
}

fn skin_vertices(model: &AnnyModel, coeffs: &[f64], bone_transforms: &[Mat4]) -> Vec<Vec3> {
    let rest_vertices = blended_points(&model.template_vertices, &model.blendshapes, coeffs);
    rest_vertices
        .iter()
        .enumerate()
        .map(|(vertex, &rest)| {
            let mut out = [0.0; 3];
            for (slot, &joint) in model.lbs_joint_indices[vertex].iter().enumerate() {
                let weight = model.lbs_joint_weights[vertex][slot];
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

fn blended_points(template: &[Vec3], blendshapes: &[Vec<Vec3>], coeffs: &[f64]) -> Vec<Vec3> {
    let mut points = template.to_vec();
    for (coeff, shape) in coeffs.iter().zip(blendshapes) {
        if *coeff == 0.0 {
            continue;
        }
        for (point, delta) in points.iter_mut().zip(shape) {
            point[0] += coeff * delta[0];
            point[1] += coeff * delta[1];
            point[2] += coeff * delta[2];
        }
    }
    points
}

fn pack_pose(params: &AnnyParams) -> Vec<Vec3> {
    let mut pose = Vec::with_capacity(163);
    pose.push(params.global_rotation);
    pose.extend_from_slice(&params.body_pose[..54]);
    pose.extend_from_slice(&params.hand_pose[..19]);
    pose.extend_from_slice(&params.body_pose[54..61]);
    pose.extend_from_slice(&params.hand_pose[19..]);
    pose.extend_from_slice(&params.body_pose[61..]);
    pose.extend_from_slice(&params.head_pose);
    pose
}
