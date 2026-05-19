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
    Smpl(SmplParams),
    Mhr(MhrParams),
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
