use anyhow::{Context, Result};
use body_models_viser::{ModelOutput, load_json, run_fixture};
use std::path::{Path, PathBuf};

fn max_abs_diff(a: &serde_json::Value, b: &serde_json::Value) -> f64 {
    match (a, b) {
        (serde_json::Value::Number(x), serde_json::Value::Number(y)) => {
            (x.as_f64().unwrap() - y.as_f64().unwrap()).abs()
        }
        (serde_json::Value::Array(xs), serde_json::Value::Array(ys)) => xs
            .iter()
            .zip(ys)
            .map(|(x, y)| max_abs_diff(x, y))
            .fold(0.0, f64::max),
        (serde_json::Value::Object(xs), serde_json::Value::Object(ys)) => xs
            .iter()
            .filter(|(key, _)| key.as_str() == "skeleton" || key.as_str() == "mesh")
            .map(|(key, x)| max_abs_diff(x, &ys[key]))
            .fold(0.0, f64::max),
        _ => 0.0,
    }
}

fn check_case(root: &Path, model: &str, name: &str, tolerance: f64) -> Result<()> {
    let fixture_path = root
        .join("fixtures")
        .join(model)
        .join(format!("{name}.json"));
    let reference_path = root
        .join("generated")
        .join("reference")
        .join(model)
        .join(format!("{name}.json"));
    let model_data = root.join("generated").join("model_data");

    let actual = run_fixture(&model_data, &fixture_path)
        .with_context(|| format!("running fixture {}", fixture_path.display()))?;
    let expected: ModelOutput = load_json(&reference_path)
        .with_context(|| format!("loading reference {}", reference_path.display()))?;

    assert_eq!(actual.model, expected.model);
    assert_eq!(actual.case, expected.case);
    assert_eq!(actual.skeleton.len(), expected.skeleton.len());
    assert_eq!(actual.mesh.len(), expected.mesh.len());

    let actual_json = serde_json::to_value(actual)?;
    let expected_json = serde_json::to_value(expected)?;
    let diff = max_abs_diff(&actual_json, &expected_json);
    assert!(
        diff <= tolerance,
        "{model}/{name} max abs diff {diff} > {tolerance}"
    );
    Ok(())
}

#[test]
fn smpl_matches_python_reference() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for name in ["rest", "shape_pose", "translation"] {
        check_case(&root, "smpl", name, 5e-5)?;
    }
    Ok(())
}
