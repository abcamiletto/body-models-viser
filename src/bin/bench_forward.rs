use anyhow::Result;
use body_models_viser::{MhrModel, Params, SmplModel, load_json, mhr_forward, smpl_forward};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_data = root.join("generated/model_data");

    let smpl_model: SmplModel = load_json(&model_data.join("smpl.json"))?;
    let smpl_fixture: body_models_viser::Fixture =
        load_json(&root.join("fixtures/smpl/shape_pose.json"))?;
    let Params::Smpl(smpl_params) = smpl_fixture.params else {
        unreachable!();
    };

    let mhr_model: MhrModel = load_json(&model_data.join("mhr.json"))?;
    let mhr_fixture: body_models_viser::Fixture =
        load_json(&root.join("fixtures/mhr/shape_pose.json"))?;
    let Params::Mhr(mhr_params) = mhr_fixture.params else {
        unreachable!();
    };

    time("smpl", 200, || smpl_forward(&smpl_model, &smpl_params))?;
    time("mhr", 50, || mhr_forward(&mhr_model, &mhr_params))?;

    Ok(())
}

fn time<T>(name: &str, iters: u32, mut f: impl FnMut() -> Result<T>) -> Result<()> {
    f()?;
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(f()?);
    }
    let elapsed = start.elapsed();
    println!(
        "{name}: {:.3} ms/iter over {iters} iterations",
        elapsed.as_secs_f64() * 1000.0 / f64::from(iters)
    );
    Ok(())
}
