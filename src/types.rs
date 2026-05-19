use glam::{Mat3, Mat4, Vec3};
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
pub struct SmplhParams {
    pub shape: Vec<f32>,
    pub body_pose: Vec<Vec3>,
    pub hand_pose: Vec<Vec3>,
    pub pelvis_rotation: Vec3,
    pub global_rotation: Vec3,
    pub global_translation: Vec3,
}

#[derive(Debug, Deserialize)]
pub struct SmplxParams {
    pub shape: Vec<f32>,
    pub body_pose: Vec<Vec3>,
    pub hand_pose: Vec<Vec3>,
    pub head_pose: Vec<Vec3>,
    pub expression: Vec<f32>,
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

#[derive(Debug, Deserialize)]
pub struct AnnyParams {
    pub gender: f32,
    pub age: f32,
    pub muscle: f32,
    pub weight: f32,
    pub height: f32,
    pub proportions: f32,
    pub body_pose: Vec<Vec3>,
    pub head_pose: Vec<Vec3>,
    pub hand_pose: Vec<Vec3>,
    pub global_rotation: Vec3,
    pub global_translation: Vec3,
}

#[derive(Debug, Deserialize)]
pub struct SomaParams {
    pub body_pose: Vec<Vec3>,
    pub head_pose: Vec<Vec3>,
    pub hand_pose: Vec<Vec3>,
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
pub struct SmplFamilyModel {
    pub v_template: Vec<Vec3>,
    pub faces: Vec<[usize; 3]>,
    pub lbs_weights: Vec<Vec<f32>>,
    pub shapedirs: Vec<Mat3x10>,
    #[serde(default)]
    pub exprdirs: Vec<Vec<Vec<f32>>>,
    pub posedirs: Vec<Vec<f32>>,
    pub j_template: Vec<Vec3>,
    pub j_shapedirs: Vec<Mat3x10>,
    #[serde(default)]
    pub j_exprdirs: Vec<Vec<Vec<f32>>>,
    #[serde(default)]
    pub hand_mean: Vec<Vec<f32>>,
    pub parents: Vec<isize>,
}

pub type SmplModel = SmplFamilyModel;
pub type SmplhModel = SmplFamilyModel;
pub type SmplxModel = SmplFamilyModel;

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

#[derive(Debug, Deserialize)]
pub struct AnnyModel {
    pub template_vertices: Vec<Vec3>,
    pub blendshapes: Vec<Vec<Vec3>>,
    pub template_bone_heads: Vec<Vec3>,
    pub template_bone_tails: Vec<Vec3>,
    pub bone_heads_blendshapes: Vec<Vec<Vec3>>,
    pub bone_tails_blendshapes: Vec<Vec<Vec3>>,
    pub bone_rolls_rotmat: Vec<Mat3>,
    pub phenotype_mask: Vec<Vec<f32>>,
    pub lbs_joint_indices: Vec<Vec<usize>>,
    pub lbs_joint_weights: Vec<Vec<f32>>,
    pub faces: Vec<[usize; 4]>,
    pub parents: Vec<isize>,
}

#[derive(Debug, Deserialize)]
pub struct SomaModel {
    pub bind_shape_active: Vec<Vec3>,
    pub world_bind_pose: Vec<Mat4>,
    pub inverse_world_bind_pose: Vec<Mat4>,
    pub t_pose_world: Vec<Mat4>,
    pub corrective_bindpose: Vec<Mat3>,
    #[serde(rename = "corrective_W1")]
    pub corrective_w1: Vec<Vec<f32>>,
    #[serde(rename = "corrective_W2_rows")]
    pub corrective_w2_rows: Vec<usize>,
    #[serde(rename = "corrective_W2_cols")]
    pub corrective_w2_cols: Vec<usize>,
    #[serde(rename = "corrective_W2_values")]
    pub corrective_w2_values: Vec<f32>,
    pub skin_joint_indices: Vec<Vec<isize>>,
    pub skin_joint_weights: Vec<Vec<f32>>,
    pub faces: Vec<[usize; 3]>,
    pub parents: Vec<isize>,
}
