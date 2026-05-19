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
