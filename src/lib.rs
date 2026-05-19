mod anny;
mod garment;
mod math;
mod mhr;
mod smpl;
mod soma;
mod types;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::fs;
use std::path::Path;

pub use anny::anny_forward;
pub use garment::garment_forward;
pub use mhr::mhr_forward;
pub use smpl::{smpl_forward, smplh_forward, smplx_forward};
pub use soma::soma_forward;
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
        ("smplh", Params::Smplh(params)) => {
            let model_data = load_json(&model_data_dir.join("smplh.json"))?;
            smpl::smplh_forward(&model_data, &params)?
        }
        ("smplx", Params::Smplx(params)) => {
            let model_data = load_json(&model_data_dir.join("smplx.json"))?;
            smpl::smplx_forward(&model_data, &params)?
        }
        ("mhr", Params::Mhr(params)) => {
            let model_data = load_json(&model_data_dir.join("mhr.json"))?;
            mhr::mhr_forward(&model_data, &params)?
        }
        ("anny", Params::Anny(params)) => {
            let model_data = load_json(&model_data_dir.join("anny.json"))?;
            anny::anny_forward(&model_data, &params)?
        }
        ("soma", Params::Soma(params)) => {
            let model_data = load_json(&model_data_dir.join("soma.json"))?;
            soma::soma_forward(&model_data, &params)?
        }
        ("garment", Params::Garment(params)) => {
            let model_data = load_json(&model_data_dir.join("garment.json"))?;
            garment::garment_forward(&model_data, &params)?
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
