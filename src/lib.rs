mod math;
mod mhr;
mod smpl;
mod types;

use anyhow::{Context, Result, bail};
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

pub fn run_fixture(model_data_dir: &Path, fixture_path: &Path) -> Result<ModelOutput> {
    let Fixture {
        model,
        case,
        params,
    } = load_json(fixture_path)?;

    let (skeleton, mesh) = match (model.as_str(), params) {
        ("smpl", Params::Smpl(params)) => {
            let model_data = load_json(&model_data_dir.join("smpl.json"))?;
            smpl::smpl_forward(&model_data, &params)?
        }
        ("mhr", Params::Mhr(params)) => {
            let model_data = load_json(&model_data_dir.join("mhr.json"))?;
            mhr::mhr_forward(&model_data, &params)?
        }
        (model, _) => bail!("unsupported or mismatched fixture model {model:?}"),
    };

    Ok(ModelOutput {
        model,
        case,
        skeleton,
        mesh,
    })
}
