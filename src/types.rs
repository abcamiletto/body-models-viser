use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub model: String,
    pub case: String,
    pub params: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelOutput {
    pub model: String,
    pub case: String,
    pub skeleton: Vec<Mat4>,
    pub mesh: Vec<Vec3>,
}
