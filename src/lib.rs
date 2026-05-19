mod mhr;
mod smpl;
mod types;

use anyhow::{Context, Result, bail};
use glam::{Mat4, Quat, Vec3};
use serde::Deserialize;
use std::fs;
use std::path::Path;

pub use mhr::mhr_forward;
pub use smpl::smpl_forward;
pub use types::*;

pub fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub(crate) fn axis_angle_rigid_transform(rotation: Vec3, translation: Vec3) -> Mat4 {
    Mat4::from_rotation_translation(
        Quat::from_axis_angle(rotation.normalize_or_zero(), rotation.length()),
        translation,
    )
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
        model => bail!("unsupported fixture model {model:?}"),
    };

    Ok(ModelOutput {
        model,
        case,
        skeleton,
        mesh,
    })
}
