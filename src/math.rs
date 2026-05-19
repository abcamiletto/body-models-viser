use anyhow::{Result, bail};
use glam::{DMat3, DMat4, DQuat, DVec3};
use std::sync::OnceLock;

use crate::types::{Mat3, Mat4, SparseRows, Vec3};

pub(crate) fn local_offsets(joints: &[Vec3], parents: &[isize]) -> Vec<Vec3> {
    joints
        .iter()
        .enumerate()
        .map(|(i, &joint)| {
            if parents[i] < 0 {
                joint
            } else {
                sub3(joint, joints[parents[i] as usize])
            }
        })
        .collect()
}

pub(crate) fn fk(rotations: &[Mat3], translations: &[Vec3], parents: &[isize]) -> Vec<Mat4> {
    let local: Vec<Mat4> = rotations
        .iter()
        .zip(translations)
        .map(|(&r, &t)| rt_to_mat4(r, t))
        .collect();
    let mut world = vec![[[0.0; 4]; 4]; rotations.len()];
    for i in 0..rotations.len() {
        world[i] = if parents[i] < 0 {
            local[i]
        } else {
            mat4_mul(world[parents[i] as usize], local[i])
        };
    }
    world
}

pub(crate) fn apply_global_skeleton(
    skeleton: &[Mat4],
    rotation: Vec3,
    translation: Vec3,
) -> Vec<Mat4> {
    let global = rt_to_mat4(axis_angle_to_mat3(rotation), translation);
    skeleton
        .iter()
        .copied()
        .map(|transform| mat4_mul(global, transform))
        .collect()
}

pub(crate) fn apply_global_points(points: &mut [Vec3], rotation: Vec3, translation: Vec3) {
    let r = to_dmat3(axis_angle_to_mat3(rotation));
    let t = to_dvec3(translation);
    for point in points {
        *point = from_dvec3(r * to_dvec3(*point) + t);
    }
}

pub(crate) fn ensure_len<T>(values: &[T], len: usize, name: &str) -> Result<()> {
    if values.len() != len {
        bail!("expected {name} length {len}, got {}", values.len());
    }
    Ok(())
}

pub(crate) fn dot<const N: usize>(a: &[f64; N], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn sparse_rows<'a>(
    cache: &'a OnceLock<SparseRows<f64>>,
    dense: &[Vec<f64>],
) -> &'a SparseRows<f64> {
    cache.get_or_init(|| {
        let mut offsets = Vec::with_capacity(dense.len() + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        offsets.push(0);
        for row in dense {
            for (index, value) in row.iter().copied().enumerate() {
                if value != 0.0 {
                    indices.push(index);
                    values.push(value);
                }
            }
            offsets.push(indices.len());
        }
        SparseRows {
            offsets,
            indices,
            values,
        }
    })
}

pub(crate) fn sparse_vec3_rows<'a>(
    cache: &'a OnceLock<SparseRows<Vec3>>,
    dense: &[Vec<Vec3>],
) -> &'a SparseRows<Vec3> {
    cache.get_or_init(|| {
        let mut offsets = Vec::with_capacity(dense.len() + 1);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        offsets.push(0);
        for row in dense {
            for (index, value) in row.iter().copied().enumerate() {
                if value != [0.0; 3] {
                    indices.push(index);
                    values.push(value);
                }
            }
            offsets.push(indices.len());
        }
        SparseRows {
            offsets,
            indices,
            values,
        }
    })
}

pub(crate) fn sparse_dot(rows: &SparseRows<f64>, row: usize, values: &[f64]) -> f64 {
    rows.row(row).map(|(i, value)| value * values[i]).sum()
}

pub(crate) fn axis_angle_to_mat3(v: Vec3) -> Mat3 {
    let axis_angle = to_dvec3(v);
    let theta = axis_angle.length();
    if theta < 1e-12 {
        return eye3();
    }
    from_dmat3(DMat3::from_quat(DQuat::from_axis_angle(
        axis_angle / theta,
        theta,
    )))
}

pub(crate) fn euler_xyz_to_mat3(v: Vec3) -> Mat3 {
    let (sx, cx) = v[0].sin_cos();
    let (sy, cy) = v[1].sin_cos();
    let (sz, cz) = v[2].sin_cos();
    mat3_mul(
        mat3_mul(
            [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
            [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        ),
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
    )
}

pub(crate) fn euler_xyz_to_quat(v: Vec3) -> [f64; 4] {
    let (sx, cx) = (v[0] * 0.5).sin_cos();
    let (sy, cy) = (v[1] * 0.5).sin_cos();
    let (sz, cz) = (v[2] * 0.5).sin_cos();
    [
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz,
    ]
}

pub(crate) fn quat_mul_xyzw(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

pub(crate) fn quat_xyzw_to_mat3(q: [f64; 4]) -> Mat3 {
    from_dmat3(DMat3::from_quat(
        DQuat::from_xyzw(q[0], q[1], q[2], q[3]).normalize(),
    ))
}

pub(crate) fn eye3() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

pub(crate) fn rt_to_mat4(r: Mat3, t: Vec3) -> Mat4 {
    [
        [r[0][0], r[0][1], r[0][2], t[0]],
        [r[1][0], r[1][1], r[1][2], t[1]],
        [r[2][0], r[2][1], r[2][2], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub(crate) fn invert_rigid(m: Mat4) -> Mat4 {
    from_dmat4(to_dmat4(m).inverse())
}

pub(crate) fn trs_to_mat4(t: Vec3, r: Mat3, s: f64) -> Mat4 {
    rt_to_mat4(scale_mat3(r, s), t)
}

pub(crate) fn mat4_rot(m: &Mat4) -> Mat3 {
    [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ]
}

pub(crate) fn mat4_trans(m: &Mat4) -> Vec3 {
    [m[0][3], m[1][3], m[2][3]]
}

pub(crate) fn mat3_transpose(m: Mat3) -> Mat3 {
    from_dmat3(to_dmat3(m).transpose())
}

pub(crate) fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
    from_dmat3(to_dmat3(a) * to_dmat3(b))
}

pub(crate) fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    from_dmat4(to_dmat4(a) * to_dmat4(b))
}

pub(crate) fn mat3_vec(m: Mat3, v: Vec3) -> Vec3 {
    from_dvec3(to_dmat3(m) * to_dvec3(v))
}

pub(crate) fn scale_mat3(m: Mat3, s: f64) -> Mat3 {
    from_dmat3(to_dmat3(m) * s)
}

pub(crate) fn add3(a: Vec3, b: Vec3) -> Vec3 {
    from_dvec3(to_dvec3(a) + to_dvec3(b))
}

pub(crate) fn sub3(a: Vec3, b: Vec3) -> Vec3 {
    from_dvec3(to_dvec3(a) - to_dvec3(b))
}

pub(crate) fn scale3(a: Vec3, s: f64) -> Vec3 {
    from_dvec3(to_dvec3(a) * s)
}

pub(crate) fn cross3(a: Vec3, b: Vec3) -> Vec3 {
    from_dvec3(to_dvec3(a).cross(to_dvec3(b)))
}

pub(crate) fn norm3(a: Vec3) -> f64 {
    to_dvec3(a).length()
}

fn to_dvec3(v: Vec3) -> DVec3 {
    DVec3::new(v[0], v[1], v[2])
}

fn from_dvec3(v: DVec3) -> Vec3 {
    [v.x, v.y, v.z]
}

fn to_dmat3(m: Mat3) -> DMat3 {
    DMat3::from_cols(
        DVec3::new(m[0][0], m[1][0], m[2][0]),
        DVec3::new(m[0][1], m[1][1], m[2][1]),
        DVec3::new(m[0][2], m[1][2], m[2][2]),
    )
}

fn from_dmat3(m: DMat3) -> Mat3 {
    let x = m.x_axis;
    let y = m.y_axis;
    let z = m.z_axis;
    [[x.x, y.x, z.x], [x.y, y.y, z.y], [x.z, y.z, z.z]]
}

fn to_dmat4(m: Mat4) -> DMat4 {
    DMat4::from_cols_array(&[
        m[0][0], m[1][0], m[2][0], m[3][0], m[0][1], m[1][1], m[2][1], m[3][1], m[0][2], m[1][2],
        m[2][2], m[3][2], m[0][3], m[1][3], m[2][3], m[3][3],
    ])
}

fn from_dmat4(m: DMat4) -> Mat4 {
    let a = m.to_cols_array();
    [
        [a[0], a[4], a[8], a[12]],
        [a[1], a[5], a[9], a[13]],
        [a[2], a[6], a[10], a[14]],
        [a[3], a[7], a[11], a[15]],
    ]
}
