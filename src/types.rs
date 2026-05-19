use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];
pub type Mat4 = [[f64; 4]; 4];
pub type Mat10 = [[f64; 10]; 3];
type SparseScalarRows = SparseRows<f64>;
type SparseVec3Rows = SparseRows<Vec3>;

#[derive(Debug)]
pub(crate) struct SparseRows<T> {
    pub(crate) offsets: Vec<usize>,
    pub(crate) indices: Vec<usize>,
    pub(crate) values: Vec<T>,
}

impl<T: Copy> SparseRows<T> {
    pub(crate) fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub(crate) fn row(&self, index: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        let start = self.offsets[index];
        let end = self.offsets[index + 1];
        self.indices[start..end]
            .iter()
            .copied()
            .zip(self.values[start..end].iter().copied())
    }
}

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub model: String,
    pub case: String,
    pub params: Params,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Params {
    Garment(GarmentParams),
    Smpl(SmplParams),
    Mhr(MhrParams),
    Anny(AnnyParams),
    Soma(SomaParams),
}

#[derive(Debug, Deserialize)]
pub struct SmplParams {
    pub shape: Vec<f64>,
    pub body_pose: Vec<Vec3>,
    pub pelvis_rotation: Vec3,
    pub global_rotation: Vec3,
    pub global_translation: Vec3,
}

#[derive(Debug, Deserialize)]
pub struct MhrParams {
    pub shape: Vec<f64>,
    pub body_pose: Vec<f64>,
    pub hand_pose: Vec<f64>,
    pub expression: Vec<f64>,
    pub global_rotation: Vec3,
    pub global_translation: Vec3,
}

#[derive(Debug, Deserialize)]
pub struct AnnyParams {
    pub gender: f64,
    pub age: f64,
    pub muscle: f64,
    pub weight: f64,
    pub height: f64,
    pub proportions: f64,
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

#[derive(Debug, Deserialize)]
pub struct GarmentParams {
    pub shape: Vec<f64>,
    pub body_pose: Vec<Vec3>,
    pub head_pose: Vec<Vec3>,
    pub hand_pose: Vec<Vec3>,
    pub pelvis_rotation: Vec3,
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
    pub lbs_weights: Vec<Vec<f64>>,
    pub shapedirs: Vec<Mat10>,
    pub posedirs: Vec<Vec<f64>>,
    pub j_template: Vec<Vec3>,
    pub j_shapedirs: Vec<Mat10>,
    pub parents: Vec<isize>,
    #[serde(skip)]
    pub(crate) lbs_weights_sparse: OnceLock<SparseScalarRows>,
}

#[derive(Debug, Deserialize)]
pub struct MhrModel {
    pub base_vertices: Vec<Vec3>,
    pub blendshape_dirs: Vec<Vec<Vec3>>,
    pub skin_weights: Vec<Vec<f64>>,
    pub skin_indices: Vec<Vec<usize>>,
    pub faces: Vec<[usize; 3]>,
    pub joint_offsets: Vec<Vec3>,
    pub joint_pre_rotations: Vec<[f64; 4]>,
    pub parameter_transform: Vec<Vec<f64>>,
    pub bind_inv_linear: Vec<Mat3>,
    pub bind_inv_translation: Vec<Vec3>,
    #[serde(rename = "corrective_W1")]
    pub corrective_w1: Vec<Vec<f64>>,
    #[serde(rename = "corrective_W2")]
    pub corrective_w2: Vec<Vec<f64>>,
    pub parents: Vec<isize>,
    #[serde(skip)]
    pub(crate) blendshape_dirs_sparse: OnceLock<SparseVec3Rows>,
    #[serde(skip)]
    pub(crate) parameter_transform_sparse: OnceLock<SparseScalarRows>,
    #[serde(skip)]
    pub(crate) corrective_w1_sparse: OnceLock<SparseScalarRows>,
    #[serde(skip)]
    pub(crate) corrective_w2_sparse: OnceLock<SparseScalarRows>,
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
    pub phenotype_mask: Vec<Vec<f64>>,
    pub lbs_joint_indices: Vec<Vec<usize>>,
    pub lbs_joint_weights: Vec<Vec<f64>>,
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
    pub corrective_w1: Vec<Vec<f64>>,
    #[serde(rename = "corrective_W2_rows")]
    pub corrective_w2_rows: Vec<usize>,
    #[serde(rename = "corrective_W2_cols")]
    pub corrective_w2_cols: Vec<usize>,
    #[serde(rename = "corrective_W2_values")]
    pub corrective_w2_values: Vec<f64>,
    pub skin_joint_indices: Vec<Vec<isize>>,
    pub skin_joint_weights: Vec<Vec<f64>>,
    pub faces: Vec<[usize; 3]>,
    pub parents: Vec<isize>,
}

#[derive(Debug, Deserialize)]
pub struct GarmentModel {
    pub mean_vertices: Vec<Vec3>,
    pub components: Vec<Mat3x15>,
    pub eigenvalues: Vec<f64>,
    pub bind_quats: Vec<[f64; 4]>,
    pub skin_joint_indices: Vec<Vec<usize>>,
    pub skin_joint_weights: Vec<Vec<f64>>,
    pub mvc_weights: Vec<Vec<f64>>,
    pub faces: Vec<[usize; 3]>,
    pub parents: Vec<isize>,
}

pub type Mat3x15 = [[f64; 15]; 3];
