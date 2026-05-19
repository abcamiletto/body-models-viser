use anyhow::Result;
use body_models_viser::{
    AnnyModel, GarmentModel, MhrModel, Params, SmplModel, SomaModel, anny_forward, garment_forward,
    load_json, mhr_forward, smpl_forward, soma_forward,
};
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

    let anny_model: AnnyModel = load_json(&model_data.join("anny.json"))?;
    let anny_fixture: body_models_viser::Fixture =
        load_json(&root.join("fixtures/anny/shape_pose.json"))?;
    let Params::Anny(anny_params) = anny_fixture.params else {
        unreachable!();
    };

    let soma_model: SomaModel = load_json(&model_data.join("soma.json"))?;
    let soma_fixture: body_models_viser::Fixture =
        load_json(&root.join("fixtures/soma/shape_pose.json"))?;
    let Params::Soma(soma_params) = soma_fixture.params else {
        unreachable!();
    };

    let garment_model: GarmentModel = load_json(&model_data.join("garment.json"))?;
    let garment_fixture: body_models_viser::Fixture =
        load_json(&root.join("fixtures/garment/shape_pose.json"))?;
    let Params::Garment(garment_params) = garment_fixture.params else {
        unreachable!();
    };

    time("smpl", 200, || smpl_forward(&smpl_model, &smpl_params))?;
    time("mhr", 50, || mhr_forward(&mhr_model, &mhr_params))?;
    time("anny", 20, || anny_forward(&anny_model, &anny_params))?;
    time("soma", 20, || soma_forward(&soma_model, &soma_params))?;
    time("garment", 20, || {
        garment_forward(&garment_model, &garment_params)
    })?;

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
