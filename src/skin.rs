use glam::{Mat4, Vec3};

pub struct ForwardInputs<'a> {
    pub lbs_weights: &'a [f32],
    pub rest_vertices: &'a [f32],
    pub skinning_transforms: &'a [f32],
    pub pose_offsets: &'a [f32],
    pub global_rotation: &'a [f32],
    pub global_translation: &'a [f32],
}

pub fn forward_vertices(inputs: ForwardInputs<'_>, output_vertices: &mut [f32]) {
    let bone_count = inputs.skinning_transforms.len() / 16;
    let skinning_transforms = (0..bone_count)
        .map(|bone| {
            let start = 16 * bone;
            mat4_from_rows(&inputs.skinning_transforms[start..start + 16])
        })
        .collect::<Vec<_>>();
    skin_with_transforms(
        inputs.lbs_weights,
        inputs.rest_vertices,
        &skinning_transforms,
        inputs.pose_offsets,
        inputs.global_rotation,
        inputs.global_translation,
        output_vertices,
    );
}

fn skin_with_transforms(
    lbs_weights: &[f32],
    rest_vertices: &[f32],
    skinning_transforms: &[Mat4],
    pose_offsets: &[f32],
    global_rotation: &[f32],
    global_translation: &[f32],
    output_vertices: &mut [f32],
) {
    let bone_count = skinning_transforms.len();
    let vertex_count = rest_vertices.len() / 3;
    let global_rotation = vec3(global_rotation, 0);
    let global = Mat4::from_rotation_translation(
        glam::Quat::from_axis_angle(
            global_rotation.normalize_or_zero(),
            global_rotation.length(),
        ),
        vec3(global_translation, 0),
    );

    for vertex in 0..vertex_count {
        let point = vec3(rest_vertices, vertex) + vec3(pose_offsets, vertex);
        let weights = &lbs_weights[vertex * bone_count..(vertex + 1) * bone_count];
        let mut transform = Mat4::ZERO;
        for (joint, &weight) in weights.iter().enumerate() {
            transform += skinning_transforms[joint] * weight;
        }
        let output = global.transform_point3(transform.transform_point3(point));
        output_vertices[3 * vertex] = output.x;
        output_vertices[3 * vertex + 1] = output.y;
        output_vertices[3 * vertex + 2] = output.z;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_vertices_applies_pose_and_global_transform() {
        let inputs = ForwardInputs {
            lbs_weights: &[1.0],
            rest_vertices: &[1.0, 2.0, 3.0],
            skinning_transforms: &[
                1.0, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 1.0, 6.0, 0.0, 0.0, 0.0, 1.0,
            ],
            pose_offsets: &[0.5, 1.0, 1.5],
            global_rotation: &[0.0, 0.0, 0.0],
            global_translation: &[10.0, 20.0, 30.0],
        };
        let mut output = [0.0; 3];

        forward_vertices(inputs, &mut output);

        assert_eq!(output, [15.5, 28.0, 40.5]);
    }
}
