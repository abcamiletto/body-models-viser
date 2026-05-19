mod anny;
mod mhr;
mod smpl;
mod soma;
mod types;

use anyhow::{Context, Result, bail};
use glam::{Mat3, Mat4, Quat, Vec3};
use serde::Deserialize;
use std::fs;
use std::path::Path;

pub use anny::anny_forward;
pub use mhr::mhr_forward;
pub use smpl::{smpl_forward, smplh_forward, smplx_forward};
pub use soma::soma_forward;
pub use types::*;

pub fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub(crate) fn axis_angle_rigid_transform(rotation: Vec3, translation: Vec3) -> Mat4 {
    Mat4::from_rotation_translation(axis_angle_quat(rotation), translation)
}

pub(crate) fn axis_angle_quat(rotation: Vec3) -> Quat {
    Quat::from_axis_angle(rotation.normalize_or_zero(), rotation.length())
}

pub(crate) fn mat4_from_mat3_translation(linear: Mat3, translation: Vec3) -> Mat4 {
    Mat4::from_cols(
        linear.x_axis.extend(0.0),
        linear.y_axis.extend(0.0),
        linear.z_axis.extend(0.0),
        translation.extend(1.0),
    )
}

pub(crate) fn ensure_len<T>(values: &[T], len: usize, name: &str) -> Result<()> {
    anyhow::ensure!(
        values.len() == len,
        "expected {name} length {len}, got {}",
        values.len()
    );
    Ok(())
}

pub fn run_fixture(model_data_dir: &Path, fixture_path: &Path) -> Result<ModelOutput> {
    let Fixture {
        model,
        case,
        params,
    } = load_json(fixture_path)?;

    let (skeleton, mesh) = match model.as_str() {
        "smpl" => {
            let params = serde_json::from_value(params).context("parsing SMPL fixture params")?;
            let model_data = load_json(&model_data_dir.join("smpl.json"))?;
            smpl_forward(&model_data, &params)?
        }
        "mhr" => {
            let params = serde_json::from_value(params).context("parsing MHR fixture params")?;
            let model_data = load_json(&model_data_dir.join("mhr.json"))?;
            mhr_forward(&model_data, &params)?
        }
        "smplh" => {
            let params = serde_json::from_value(params).context("parsing SMPLH fixture params")?;
            let model_data = load_json(&model_data_dir.join("smplh.json"))?;
            smplh_forward(&model_data, &params)?
        }
        "smplx" => {
            let params = serde_json::from_value(params).context("parsing SMPLX fixture params")?;
            let model_data = load_json(&model_data_dir.join("smplx.json"))?;
            smplx_forward(&model_data, &params)?
        }
        "anny" => {
            let params = serde_json::from_value(params).context("parsing ANNY fixture params")?;
            let model_data = load_json(&model_data_dir.join("anny.json"))?;
            anny_forward(&model_data, &params)?
        }
        "soma" => {
            let params = serde_json::from_value(params).context("parsing SOMA fixture params")?;
            let model_data = load_json(&model_data_dir.join("soma.json"))?;
            soma_forward(&model_data, &params)?
        }
        model => bail!("unsupported fixture model {model:?}"),
    };

    Ok(ModelOutput {
        model,
        case,
        skeleton,
        mesh,
    })
}
