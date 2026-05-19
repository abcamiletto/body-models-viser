use anyhow::Result;
use glam::{Mat3, Mat4, Vec3, Vec4};

use crate::axis_angle_rigid_transform;
use crate::types::{
    SmplFamilyModel, SmplModel, SmplParams, SmplhModel, SmplhParams, SmplxModel, SmplxParams,
};

pub fn smpl_forward(model: &SmplModel, params: &SmplParams) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    ensure_len(&params.shape, 10, "SMPL shape")?;
    ensure_len(&params.body_pose, 23, "SMPL body_pose")?;

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
    ensure_len(&params.shape, 10, "SMPLH shape")?;
    ensure_len(&params.body_pose, 21, "SMPLH body_pose")?;
    ensure_len(&params.hand_pose, 30, "SMPLH hand_pose")?;

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
    ensure_len(&params.shape, 10, "SMPLX shape")?;
    ensure_len(&params.expression, 10, "SMPLX expression")?;
    ensure_len(&params.body_pose, 21, "SMPLX body_pose")?;
    ensure_len(&params.hand_pose, 30, "SMPLX hand_pose")?;
    ensure_len(&params.head_pose, 3, "SMPLX head_pose")?;

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
    shape: &[f32],
    expression: &[f32],
    pose: &[Mat3],
    global_rotation: Vec3,
    global_translation: Vec3,
) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    let joints = shaped_joints(model, shape, expression);
    let skeleton = fk(
        pose,
        &local_offsets(&joints, &model.parents),
        &model.parents,
    );
    let mut mesh = posed_vertices(model, shape, expression, pose);
    skin_vertices(model, &joints, &skeleton, &mut mesh);

    let global = axis_angle_rigid_transform(global_rotation, global_translation);
    for vertex in &mut mesh {
        *vertex = global.transform_point3(*vertex);
    }
    let skeleton = skeleton
        .into_iter()
        .map(|transform| global * transform)
        .collect();
    Ok((skeleton, mesh))
}

fn shaped_joints(model: &SmplFamilyModel, shape: &[f32], expression: &[f32]) -> Vec<Vec3> {
    model
        .j_template
        .iter()
        .zip(&model.j_shapedirs)
        .enumerate()
        .map(|(joint, (&position, dirs))| {
            position
                + blend_shape(dirs, shape)
                + expr_blend_shape(&model.j_exprdirs, joint, expression)
        })
        .collect()
}

fn posed_vertices(
    model: &SmplFamilyModel,
    shape: &[f32],
    expression: &[f32],
    pose: &[Mat3],
) -> Vec<Vec3> {
    let mut vertices: Vec<Vec3> = model
        .v_template
        .iter()
        .zip(&model.shapedirs)
        .enumerate()
        .map(|(vertex, (&position, dirs))| {
            position
                + blend_shape(dirs, shape)
                + expr_blend_shape(&model.exprdirs, vertex, expression)
        })
        .collect();

    for (delta, row) in pose_delta(pose).iter().zip(&model.posedirs) {
        if *delta == 0.0 {
            continue;
        }
        for (vertex, pose_delta) in vertices.iter_mut().zip(row.chunks_exact(3)) {
            *vertex += *delta * Vec3::new(pose_delta[0], pose_delta[1], pose_delta[2]);
        }
    }
    vertices
}

fn blend_shape(dirs: &[[f32; 10]; 3], shape: &[f32]) -> Vec3 {
    Vec3::new(
        dot(&dirs[0], shape),
        dot(&dirs[1], shape),
        dot(&dirs[2], shape),
    )
}

fn expr_blend_shape(dirs: &[Vec<Vec<f32>>], index: usize, expression: &[f32]) -> Vec3 {
    let Some(dirs) = dirs.get(index) else {
        return Vec3::ZERO;
    };
    Vec3::new(
        dot_expr(&dirs[0], expression),
        dot_expr(&dirs[1], expression),
        dot_expr(&dirs[2], expression),
    )
}

fn pose_delta(pose: &[Mat3]) -> Vec<f32> {
    pose[1..]
        .iter()
        .flat_map(|rotation| {
            let delta = rotation - Mat3::IDENTITY;
            [
                delta.x_axis.x,
                delta.y_axis.x,
                delta.z_axis.x,
                delta.x_axis.y,
                delta.y_axis.y,
                delta.z_axis.y,
                delta.x_axis.z,
                delta.y_axis.z,
                delta.z_axis.z,
            ]
        })
        .collect()
}

fn skin_vertices(
    model: &SmplFamilyModel,
    joints: &[Vec3],
    skeleton: &[Mat4],
    vertices: &mut [Vec3],
) {
    let joint_transforms: Vec<Mat4> = skeleton
        .iter()
        .zip(joints)
        .map(|(&transform, &joint)| transform * Mat4::from_translation(-joint))
        .collect();

    for (vertex, weights) in vertices.iter_mut().zip(&model.lbs_weights) {
        let mut transform = Mat4::ZERO;
        for (joint, weight) in weights.iter().copied().enumerate() {
            transform += joint_transforms[joint] * weight;
        }
        *vertex = transform.transform_point3(*vertex);
    }
}

fn local_offsets(joints: &[Vec3], parents: &[isize]) -> Vec<Vec3> {
    joints
        .iter()
        .enumerate()
        .map(|(joint, &position)| {
            if parents[joint] < 0 {
                position
            } else {
                position - joints[parents[joint] as usize]
            }
        })
        .collect()
}

fn fk(rotations: &[Mat3], translations: &[Vec3], parents: &[isize]) -> Vec<Mat4> {
    let mut world = vec![Mat4::IDENTITY; rotations.len()];
    for joint in 0..rotations.len() {
        let local = mat4_from_mat3_translation(rotations[joint], translations[joint]);
        world[joint] = if parents[joint] < 0 {
            local
        } else {
            world[parents[joint] as usize] * local
        };
    }
    world
}

fn mat4_from_mat3_translation(rotation: Mat3, translation: Vec3) -> Mat4 {
    Mat4::from_cols(
        rotation.x_axis.extend(0.0),
        rotation.y_axis.extend(0.0),
        rotation.z_axis.extend(0.0),
        Vec4::new(translation.x, translation.y, translation.z, 1.0),
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

fn dot(a: &[f32; 10], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn dot_expr(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn pose_matrices<const N: usize>(pelvis: Vec3, parts: [&[Vec3]; N]) -> Vec<Mat3> {
    let joints = parts.iter().map(|part| part.len()).sum::<usize>() + 1;
    let mut pose = Vec::with_capacity(joints);
    pose.push(Mat3::from_axis_angle(
        pelvis.normalize_or_zero(),
        pelvis.length(),
    ));
    for part in parts {
        pose.extend(
            part.iter()
                .map(|v| Mat3::from_axis_angle(v.normalize_or_zero(), v.length())),
        );
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
            Vec3::new(
                pose.x + model.hand_mean[hand][offset],
                pose.y + model.hand_mean[hand][offset + 1],
                pose.z + model.hand_mean[hand][offset + 2],
            )
        })
        .collect()
}
