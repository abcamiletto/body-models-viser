use anyhow::Result;
use glam::{Mat3, Mat4, Quat, Vec3};

use crate::types::{AnnyModel, AnnyParams};

const PHENOTYPE_ANCHORS: [&[f32]; 8] = [
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
    ensure_len(&params.body_pose, 64, "ANNY body_pose")?;
    ensure_len(&params.head_pose, 60, "ANNY head_pose")?;
    ensure_len(&params.hand_pose, 38, "ANNY hand_pose")?;

    let pose = pack_pose(params);
    let pose_transforms: Vec<Mat4> = pose
        .iter()
        .map(|&rotation| Mat4::from_quat(axis_angle_quat(rotation)))
        .collect();
    let coeffs = phenotype_coeffs(model, params);
    let rest_poses = rest_poses(model, &coeffs);
    let (mut skeleton, bone_transforms) = forward_kinematics(model, &rest_poses, &pose_transforms);
    let mut mesh = skin_vertices(model, &coeffs, &bone_transforms);

    for vertex in &mut mesh {
        *vertex += params.global_translation;
    }
    for transform in &mut skeleton {
        transform.w_axis.x += params.global_translation.x;
        transform.w_axis.y += params.global_translation.y;
        transform.w_axis.z += params.global_translation.z;
    }
    Ok((skeleton, mesh))
}

fn phenotype_coeffs(model: &AnnyModel, params: &AnnyParams) -> Vec<f32> {
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

fn interpolation_weights(value: f32, anchors: &[f32]) -> Vec<f32> {
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

fn rest_poses(model: &AnnyModel, coeffs: &[f32]) -> Vec<Mat4> {
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
        .map(|((&head, tail), &roll)| {
            mat4_from_mat3_translation(bone_rotation(head, tail, roll), head)
        })
        .collect()
}

fn bone_rotation(head: Vec3, tail: Vec3, roll: Mat3) -> Mat3 {
    let y_axis = Vec3::Y;
    let y = (tail - head).normalize();
    let cross = y.cross(y_axis);
    let cross_norm = cross.length();
    let axis = cross / cross_norm;
    let angle = cross_norm.atan2(y.dot(y_axis));
    let rotation = if (axis.length_squared() - 1.0).abs() < 0.1 {
        Mat3::from_axis_angle(axis, -angle)
    } else {
        Mat3::from_cols(Vec3::X, -Vec3::Y, -Vec3::Z)
    };
    rotation * roll
}

fn forward_kinematics(
    model: &AnnyModel,
    rest_poses: &[Mat4],
    pose: &[Mat4],
) -> (Vec<Mat4>, Vec<Mat4>) {
    let root_rest = rest_poses[0];
    let base = root_rest.inverse();
    let root_rotation = mat4_from_mat3_translation(Mat3::from_mat4(root_rest), Vec3::ZERO);
    let mut delta = pose.to_vec();
    delta[0] = pose[0] * root_rotation;

    let rest_inv: Vec<Mat4> = rest_poses
        .iter()
        .map(|transform| transform.inverse())
        .collect();
    let local: Vec<Mat4> = rest_poses
        .iter()
        .zip(delta)
        .map(|(&rest, delta)| rest * delta)
        .collect();
    let mut skeleton = vec![Mat4::IDENTITY; rest_poses.len()];
    let mut transforms = vec![Mat4::IDENTITY; rest_poses.len()];
    for joint in 0..rest_poses.len() {
        skeleton[joint] = if model.parents[joint] < 0 {
            base * local[joint]
        } else {
            transforms[model.parents[joint] as usize] * local[joint]
        };
        transforms[joint] = skeleton[joint] * rest_inv[joint];
    }
    (skeleton, transforms)
}

fn skin_vertices(model: &AnnyModel, coeffs: &[f32], bone_transforms: &[Mat4]) -> Vec<Vec3> {
    let rest_vertices = blended_points(&model.template_vertices, &model.blendshapes, coeffs);
    rest_vertices
        .iter()
        .enumerate()
        .map(|(vertex, &rest)| {
            let mut out = Vec3::ZERO;
            for (slot, &joint) in model.lbs_joint_indices[vertex].iter().enumerate() {
                let weight = model.lbs_joint_weights[vertex][slot];
                out += weight * bone_transforms[joint].transform_point3(rest);
            }
            out
        })
        .collect()
}

fn blended_points(template: &[Vec3], blendshapes: &[Vec<Vec3>], coeffs: &[f32]) -> Vec<Vec3> {
    let mut points = template.to_vec();
    for (coeff, shape) in coeffs.iter().zip(blendshapes) {
        if *coeff == 0.0 {
            continue;
        }
        for (point, &delta) in points.iter_mut().zip(shape) {
            *point += *coeff * delta;
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
