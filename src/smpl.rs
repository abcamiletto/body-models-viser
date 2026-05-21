use glam::{Mat4, Vec3};

pub struct ForwardInputs<'a> {
    pub lbs_weights: &'a [f32],
    pub rest_joints: &'a [f32],
    pub rest_vertices: &'a [f32],
    pub joint_transforms: &'a [f32],
    pub pose_offsets: &'a [f32],
    pub global_rotation: &'a [f32],
    pub global_translation: &'a [f32],
}

pub fn forward_vertices(inputs: ForwardInputs<'_>, output_vertices: &mut [f32]) {
    let joint_count = inputs.rest_joints.len() / 3;
    let vertex_count = inputs.rest_vertices.len() / 3;
    let joint_transforms = (0..joint_count)
        .map(|joint| transform_for_joint(&inputs, joint))
        .collect::<Vec<_>>();
    let global_rotation = vec3(inputs.global_rotation, 0);
    let global = Mat4::from_rotation_translation(
        glam::Quat::from_axis_angle(
            global_rotation.normalize_or_zero(),
            global_rotation.length(),
        ),
        vec3(inputs.global_translation, 0),
    );

    for vertex in 0..vertex_count {
        let point = vec3(inputs.rest_vertices, vertex) + vec3(inputs.pose_offsets, vertex);
        let weights = &inputs.lbs_weights[vertex * joint_count..(vertex + 1) * joint_count];
        let mut transform = Mat4::ZERO;
        for (joint, &weight) in weights.iter().enumerate() {
            transform += joint_transforms[joint] * weight;
        }
        let output = global.transform_point3(transform.transform_point3(point));
        output_vertices[3 * vertex] = output.x;
        output_vertices[3 * vertex + 1] = output.y;
        output_vertices[3 * vertex + 2] = output.z;
    }
}

fn transform_for_joint(inputs: &ForwardInputs<'_>, joint: usize) -> Mat4 {
    mat4_from_rows(&inputs.joint_transforms[16 * joint..16 * joint + 16])
        * Mat4::from_translation(-vec3(inputs.rest_joints, joint))
}

fn vec3(values: &[f32], index: usize) -> Vec3 {
    Vec3::new(
        values[3 * index],
        values[3 * index + 1],
        values[3 * index + 2],
    )
}

fn mat4_from_rows(rows: &[f32]) -> Mat4 {
    Mat4::from_cols_array(&[
        rows[0], rows[4], rows[8], rows[12], rows[1], rows[5], rows[9], rows[13], rows[2], rows[6],
        rows[10], rows[14], rows[3], rows[7], rows[11], rows[15],
    ])
}
