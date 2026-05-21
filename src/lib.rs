mod smpl;

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
/// All pointers must refer to valid contiguous buffers for the given lengths.
pub unsafe extern "C" fn smpl_forward_vertices(
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
    output_vertices_ptr: *mut f32,
) {
    assert_eq!(rest_joints_len % 3, 0);
    assert_eq!(rest_vertices_len % 3, 0);
    assert_eq!(joint_transforms_len % 16, 0);
    assert_eq!(pose_offsets_len, rest_vertices_len);
    assert_eq!(
        lbs_weights_len,
        (rest_vertices_len / 3) * (rest_joints_len / 3)
    );

    let vertex_count = rest_vertices_len / 3;
    smpl::forward_vertices(
        smpl::ForwardInputs {
            lbs_weights: unsafe { std::slice::from_raw_parts(lbs_weights_ptr, lbs_weights_len) },
            rest_joints: unsafe { std::slice::from_raw_parts(rest_joints_ptr, rest_joints_len) },
            rest_vertices: unsafe {
                std::slice::from_raw_parts(rest_vertices_ptr, rest_vertices_len)
            },
            joint_transforms: unsafe {
                std::slice::from_raw_parts(joint_transforms_ptr, joint_transforms_len)
            },
            pose_offsets: unsafe { std::slice::from_raw_parts(pose_offsets_ptr, pose_offsets_len) },
            global_rotation: unsafe { std::slice::from_raw_parts(global_rotation_ptr, 3) },
            global_translation: unsafe { std::slice::from_raw_parts(global_translation_ptr, 3) },
        },
        unsafe { std::slice::from_raw_parts_mut(output_vertices_ptr, vertex_count * 3) },
    );
}
