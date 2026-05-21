use glam::{Mat4, Vec3};

pub struct Model {
    pub lbs_weights: Vec<f32>,
}

pub struct Identity {
    pub rest_joints: Vec<Vec3>,
    pub rest_vertices: Vec<Vec3>,
}

pub struct Pose {
    pub joint_transforms: Vec<Mat4>,
    pub pose_offsets: Vec<Vec3>,
    pub global_rotation: Vec3,
    pub global_translation: Vec3,
}

pub fn forward_vertices(model: &Model, identity: &Identity, pose: &Pose) -> Vec<Vec3> {
    let joint_transforms = pose
        .joint_transforms
        .iter()
        .zip(&identity.rest_joints)
        .map(|(&transform, &joint)| transform * Mat4::from_translation(-joint))
        .collect::<Vec<_>>();

    let global = Mat4::from_rotation_translation(
        glam::Quat::from_axis_angle(
            pose.global_rotation.normalize_or_zero(),
            pose.global_rotation.length(),
        ),
        pose.global_translation,
    );

    identity
        .rest_vertices
        .iter()
        .zip(&pose.pose_offsets)
        .zip(model.lbs_weights.chunks_exact(pose.joint_transforms.len()))
        .map(|((&rest_vertex, &pose_offset), weights)| {
            let mut transform = Mat4::ZERO;
            for (joint, &weight) in weights.iter().enumerate() {
                transform += joint_transforms[joint] * weight;
            }
            global.transform_point3(transform.transform_point3(rest_vertex + pose_offset))
        })
        .collect()
}
