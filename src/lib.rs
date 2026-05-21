mod smpl;

use glam::{Mat4, Vec3};
use std::cell::RefCell;

struct SmplRuntimeModel {
    model: smpl::Model,
    identity: smpl::Identity,
    pose: smpl::Pose,
}

thread_local! {
    static SMPL_MODELS: RefCell<Vec<SmplRuntimeModel>> = const { RefCell::new(Vec::new()) };
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut bytes = Vec::<u8>::with_capacity(len);
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    ptr
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `ptr` must come from `alloc(len)` and must not be used after this call.
pub unsafe extern "C" fn wasm_free(ptr: *mut u8, len: usize) {
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len) });
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `handle` must come from a Rust function in this module that returns output.
pub unsafe extern "C" fn output_free(handle: u64) {
    let (ptr, len) = unpack(handle);
    drop(unsafe { Vec::from_raw_parts(ptr as *mut f32, 0, len) });
}

#[unsafe(no_mangle)]
/// # Safety
///
/// All pointers must refer to valid contiguous buffers for the given lengths.
pub unsafe extern "C" fn smpl_create(
    lbs_weights_ptr: *const f32,
    lbs_weights_len: usize,
    rest_joints_ptr: *const f32,
    rest_joints_len: usize,
    rest_vertices_ptr: *const f32,
    rest_vertices_len: usize,
    joint_transforms_ptr: *const f32,
    joint_transforms_len: usize,
    pose_offsets_ptr: *const f32,
    pose_offsets_len: usize,
    global_rotation_ptr: *const f32,
    global_translation_ptr: *const f32,
) -> usize {
    assert_eq!(rest_joints_len % 3, 0);
    assert_eq!(rest_vertices_len % 3, 0);
    assert_eq!(joint_transforms_len % 16, 0);
    assert_eq!(pose_offsets_len, rest_vertices_len);
    assert_eq!(
        lbs_weights_len,
        (rest_vertices_len / 3) * (rest_joints_len / 3)
    );

    SMPL_MODELS.with_borrow_mut(|models| {
        let id = models.len();
        models.push(SmplRuntimeModel {
            model: smpl::Model {
                lbs_weights: unsafe { read_slice(lbs_weights_ptr, lbs_weights_len) },
            },
            identity: smpl::Identity {
                rest_joints: unsafe { read_vec3s(rest_joints_ptr, rest_joints_len) },
                rest_vertices: unsafe { read_vec3s(rest_vertices_ptr, rest_vertices_len) },
            },
            pose: smpl::Pose {
                joint_transforms: unsafe { read_mat4s(joint_transforms_ptr, joint_transforms_len) },
                pose_offsets: unsafe { read_vec3s(pose_offsets_ptr, pose_offsets_len) },
                global_rotation: unsafe { read_vec3(global_rotation_ptr) },
                global_translation: unsafe { read_vec3(global_translation_ptr) },
            },
        });
        id
    })
}

#[unsafe(no_mangle)]
/// # Safety
///
/// All pointers must refer to valid contiguous buffers for the given lengths.
pub unsafe extern "C" fn smpl_set_identity(
    id: usize,
    rest_joints_ptr: *const f32,
    rest_joints_len: usize,
    rest_vertices_ptr: *const f32,
    rest_vertices_len: usize,
) {
    assert_eq!(rest_joints_len % 3, 0);
    assert_eq!(rest_vertices_len % 3, 0);
    SMPL_MODELS.with_borrow_mut(|models| {
        let runtime = models.get_mut(id).expect("invalid SMPL model id");
        runtime.identity = smpl::Identity {
            rest_joints: unsafe { read_vec3s(rest_joints_ptr, rest_joints_len) },
            rest_vertices: unsafe { read_vec3s(rest_vertices_ptr, rest_vertices_len) },
        };
    });
}

#[unsafe(no_mangle)]
/// # Safety
///
/// All pointers must refer to valid contiguous buffers for the given lengths.
pub unsafe extern "C" fn smpl_set_pose(
    id: usize,
    joint_transforms_ptr: *const f32,
    joint_transforms_len: usize,
    pose_offsets_ptr: *const f32,
    pose_offsets_len: usize,
) {
    assert_eq!(joint_transforms_len % 16, 0);
    assert_eq!(pose_offsets_len % 3, 0);
    SMPL_MODELS.with_borrow_mut(|models| {
        let runtime = models.get_mut(id).expect("invalid SMPL model id");
        runtime.pose.joint_transforms =
            unsafe { read_mat4s(joint_transforms_ptr, joint_transforms_len) };
        runtime.pose.pose_offsets = unsafe { read_vec3s(pose_offsets_ptr, pose_offsets_len) };
    });
}

#[unsafe(no_mangle)]
/// # Safety
///
/// Pointers must refer to three contiguous `f32` values.
pub unsafe extern "C" fn smpl_set_global(
    id: usize,
    global_rotation_ptr: *const f32,
    global_translation_ptr: *const f32,
) {
    SMPL_MODELS.with_borrow_mut(|models| {
        let runtime = models.get_mut(id).expect("invalid SMPL model id");
        runtime.pose.global_rotation = unsafe { read_vec3(global_rotation_ptr) };
        runtime.pose.global_translation = unsafe { read_vec3(global_translation_ptr) };
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn smpl_forward(id: usize) -> u64 {
    SMPL_MODELS.with_borrow(|models| {
        let runtime = models.get(id).expect("invalid SMPL model id");
        write_vec3s(&smpl::forward_vertices(
            &runtime.model,
            &runtime.identity,
            &runtime.pose,
        ))
    })
}

unsafe fn read_slice<T: Copy>(ptr: *const T, len: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
}

unsafe fn read_vec3(ptr: *const f32) -> Vec3 {
    let values = unsafe { std::slice::from_raw_parts(ptr, 3) };
    Vec3::new(values[0], values[1], values[2])
}

unsafe fn read_vec3s(ptr: *const f32, len: usize) -> Vec<Vec3> {
    unsafe { std::slice::from_raw_parts(ptr, len) }
        .chunks_exact(3)
        .map(|xyz| Vec3::new(xyz[0], xyz[1], xyz[2]))
        .collect()
}

unsafe fn read_mat4s(ptr: *const f32, len: usize) -> Vec<Mat4> {
    unsafe { std::slice::from_raw_parts(ptr, len) }
        .chunks_exact(16)
        .map(|rows| {
            Mat4::from_cols_array(&[
                rows[0], rows[4], rows[8], rows[12], rows[1], rows[5], rows[9], rows[13], rows[2],
                rows[6], rows[10], rows[14], rows[3], rows[7], rows[11], rows[15],
            ])
        })
        .collect()
}

fn write_vec3s(vertices: &[Vec3]) -> u64 {
    let bytes = vertices
        .iter()
        .flat_map(|vertex| [vertex.x, vertex.y, vertex.z])
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let len = bytes.len();
    let ptr = Box::into_raw(bytes) as *mut f32 as *mut u8;
    pack(ptr, len)
}

fn pack(ptr: *mut u8, len: usize) -> u64 {
    ((ptr as u64) << 32) | len as u64
}

fn unpack(handle: u64) -> (*mut u8, usize) {
    ((handle >> 32) as *mut u8, (handle & 0xffff_ffff) as usize)
}
