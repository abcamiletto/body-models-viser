use anyhow::Result;
use body_models_viser::{
    AnnyModel, AnnyParams, MhrModel, MhrParams, SmplModel, SmplParams, SmplhModel, SmplhParams,
    SmplxModel, SmplxParams, SomaModel, SomaParams, anny_forward, load_json, mhr_forward,
    smpl_forward, smplh_forward, smplx_forward, soma_forward,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model_data = root.join("generated/model_data");

    let smpl_model: SmplModel = load_json(&model_data.join("smpl.json"))?;
    let smpl_fixture: body_models_viser::Fixture =
        load_json(&root.join("fixtures/smpl/shape_pose.json"))?;
    let smpl_params: SmplParams = serde_json::from_value(smpl_fixture.params)?;

    let smplh_model: SmplhModel = load_json(&model_data.join("smplh.json"))?;
    let smplh_fixture: body_models_viser::Fixture =
        load_json(&root.join("fixtures/smplh/shape_pose.json"))?;
    let smplh_params: SmplhParams = serde_json::from_value(smplh_fixture.params)?;

    let smplx_model: SmplxModel = load_json(&model_data.join("smplx.json"))?;
    let smplx_fixture: body_models_viser::Fixture =
        load_json(&root.join("fixtures/smplx/shape_pose.json"))?;
    let smplx_params: SmplxParams = serde_json::from_value(smplx_fixture.params)?;

    let mhr_model: MhrModel = load_json(&model_data.join("mhr.json"))?;
    let mhr_fixture: body_models_viser::Fixture =
        load_json(&root.join("fixtures/mhr/shape_pose.json"))?;
    let mhr_params: MhrParams = serde_json::from_value(mhr_fixture.params)?;

    let anny_model: AnnyModel = load_json(&model_data.join("anny.json"))?;
    let anny_fixture: body_models_viser::Fixture =
        load_json(&root.join("fixtures/anny/shape_pose.json"))?;
    let anny_params: AnnyParams = serde_json::from_value(anny_fixture.params)?;

    let soma_model: SomaModel = load_json(&model_data.join("soma.json"))?;
    let soma_fixture: body_models_viser::Fixture =
        load_json(&root.join("fixtures/soma/shape_pose.json"))?;
    let soma_params: SomaParams = serde_json::from_value(soma_fixture.params)?;

    time("smpl", 200, || smpl_forward(&smpl_model, &smpl_params))?;
    time("smplh", 100, || smplh_forward(&smplh_model, &smplh_params))?;
    time("smplx", 100, || smplx_forward(&smplx_model, &smplx_params))?;
    time("mhr", 50, || mhr_forward(&mhr_model, &mhr_params))?;
    time("anny", 20, || anny_forward(&anny_model, &anny_params))?;
    time("soma", 20, || soma_forward(&soma_model, &soma_params))?;
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
