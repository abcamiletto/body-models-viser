use anyhow::Result;
use glam::{Mat3, Mat4, Vec3, Vec4};

use crate::axis_angle_rigid_transform;
use crate::types::{SmplModel, SmplParams};

pub fn smpl_forward(model: &SmplModel, params: &SmplParams) -> Result<(Vec<Mat4>, Vec<Vec3>)> {
    ensure_len(&params.shape, 10, "SMPL shape")?;
    ensure_len(&params.body_pose, 23, "SMPL body_pose")?;

    let pose: Vec<Mat3> = std::iter::once(Mat3::from_axis_angle(
        params.pelvis_rotation.normalize_or_zero(),
        params.pelvis_rotation.length(),
    ))
    .chain(
        params
            .body_pose
            .iter()
            .map(|v| Mat3::from_axis_angle(v.normalize_or_zero(), v.length())),
    )
    .collect();
    let joints = shaped_joints(model, &params.shape);
    let skeleton = fk(
        &pose,
        &local_offsets(&joints, &model.parents),
        &model.parents,
    );
    let mut mesh = posed_vertices(model, params, &pose);
    skin_vertices(model, &joints, &skeleton, &mut mesh);

    let global = axis_angle_rigid_transform(params.global_rotation, params.global_translation);
    for vertex in &mut mesh {
        *vertex = global.transform_point3(*vertex);
    }
    let skeleton = skeleton
        .into_iter()
        .map(|transform| global * transform)
        .collect();
    Ok((skeleton, mesh))
}

fn shaped_joints(model: &SmplModel, shape: &[f32]) -> Vec<Vec3> {
    model
        .j_template
        .iter()
        .zip(&model.j_shapedirs)
        .map(|(&joint, dirs)| joint + blend_shape(dirs, shape))
        .collect()
}

fn posed_vertices(model: &SmplModel, params: &SmplParams, pose: &[Mat3]) -> Vec<Vec3> {
    let mut vertices: Vec<Vec3> = model
        .v_template
        .iter()
        .zip(&model.shapedirs)
        .map(|(&vertex, dirs)| vertex + blend_shape(dirs, &params.shape))
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

fn skin_vertices(model: &SmplModel, joints: &[Vec3], skeleton: &[Mat4], vertices: &mut [Vec3]) {
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
