use anyhow::{Result, bail};
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
    let r = axis_angle_to_mat3(rotation);
    for point in points {
        *point = add3(mat3_vec(r, *point), translation);
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
    let theta = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if theta < 1e-12 {
        return eye3();
    }
    let [x, y, z] = [v[0] / theta, v[1] / theta, v[2] / theta];
    let c = theta.cos();
    let s = theta.sin();
    let one_c = 1.0 - c;
    [
        [
            c + x * x * one_c,
            x * y * one_c - z * s,
            x * z * one_c + y * s,
        ],
        [
            y * x * one_c + z * s,
            c + y * y * one_c,
            y * z * one_c - x * s,
        ],
        [
            z * x * one_c - y * s,
            z * y * one_c + x * s,
            c + z * z * one_c,
        ],
    ]
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
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let [x, y, z, w] = [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
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

pub(crate) fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

pub(crate) fn mat3_vec(m: Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub(crate) fn scale_mat3(m: Mat3, s: f64) -> Mat3 {
    [
        [m[0][0] * s, m[0][1] * s, m[0][2] * s],
        [m[1][0] * s, m[1][1] * s, m[1][2] * s],
        [m[2][0] * s, m[2][1] * s, m[2][2] * s],
    ]
}

pub(crate) fn add3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn scale3(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
