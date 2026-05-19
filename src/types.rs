use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

pub type Mat3x10 = [[f32; 10]; 3];

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub model: String,
    pub case: String,
    pub params: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct SmplParams {
    pub shape: Vec<f32>,
    pub body_pose: Vec<Vec3>,
    pub pelvis_rotation: Vec3,
    pub global_rotation: Vec3,
    pub global_translation: Vec3,
}

#[derive(Debug, Deserialize)]
pub struct MhrParams {
    pub shape: Vec<f32>,
    pub body_pose: Vec<f32>,
    pub hand_pose: Vec<f32>,
    pub expression: Vec<f32>,
    pub global_rotation: Vec3,
    pub global_translation: Vec3,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelOutput {
    pub model: String,
    pub case: String,
    pub skeleton: Vec<Mat4>,
    pub mesh: Vec<Vec3>,
}

#[derive(Debug, Deserialize)]
pub struct SmplModel {
    pub v_template: Vec<Vec3>,
    pub faces: Vec<[usize; 3]>,
    pub lbs_weights: Vec<Vec<f32>>,
    pub shapedirs: Vec<Mat3x10>,
    pub posedirs: Vec<Vec<f32>>,
    pub j_template: Vec<Vec3>,
    pub j_shapedirs: Vec<Mat3x10>,
    pub parents: Vec<isize>,
}

#[derive(Debug, Deserialize)]
pub struct MhrModel {
    pub base_vertices: Vec<Vec3>,
    pub blendshape_dirs: Vec<Vec<Vec3>>,
    pub skin_weights: Vec<Vec<f32>>,
    pub skin_indices: Vec<Vec<usize>>,
    pub faces: Vec<[usize; 3]>,
    pub joint_offsets: Vec<Vec3>,
    pub joint_pre_rotations: Vec<[f32; 4]>,
    pub parameter_transform: Vec<Vec<f32>>,
    pub bind_inv_linear: Vec<[[f32; 3]; 3]>,
    pub bind_inv_translation: Vec<Vec3>,
    #[serde(rename = "corrective_W1")]
    pub corrective_w1: Vec<Vec<f32>>,
    #[serde(rename = "corrective_W2")]
    pub corrective_w2: Vec<Vec<f32>>,
    pub parents: Vec<isize>,
}
