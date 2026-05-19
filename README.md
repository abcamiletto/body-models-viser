# body-models-viser

`body-models-viser` is a browser-oriented implementation of selected
`body-models` models. The goal is to evaluate body models directly in a web
viewer, without being limited by viser skinning constraints such as a fixed
maximum number of bones per vertex.

The package currently supports unbatched/default SMPL and MHR models. The Rust
implementation is checked against Python `body-models` output using JSON
fixtures, and the Python wheel ships the JavaScript bundle that can be injected
into a viser frontend.

## Layout

- `src/` contains the Rust model implementation and JSON CLI.
- `fixtures/` contains small input cases for each model.
- `scripts/generate_reference.py` exports model weights and Python reference
  outputs into `generated/`.
- `tests/parity.rs` compares Rust output against the generated Python outputs.
- `client/` contains the TypeScript viser-facing plugin contract.
- `scripts/compare_viser_plugins.py` opens a side-by-side viser scene comparing
  the original Python plugin with this Rust-backed output.

`generated/`, `target/`, `client/dist/`, and `client/node_modules/` are local
build artifacts and are ignored.

## How It Works

Reference data is generated from a local `body-models` checkout. The script
saves two kinds of JSON:

- `generated/model_data/<model>.json`: model weights needed by Rust.
- `generated/reference/<model>/<case>.json`: expected `skeleton` and `mesh`
  outputs for each fixture.

Rust loads the same fixture JSON as Python, evaluates the model, and emits:

```json
{
  "model": "smpl",
  "case": "rest",
  "skeleton": [],
  "mesh": []
}
```

The TypeScript package mirrors the shape of the existing
`body_models.extras.viser_plugin.add_body_model()` handle. It is intentionally
small: callers provide a model object with `getRestPose`, `getBindParams`,
`forward`, and `forwardSkeleton`, and the plugin wires that into a viser-like
scene adapter.

## User Usage

Install the Python package:

```sh
uv add body-models-viser
```

Get the bundled browser module from Python:

```py
from body_models_viser import client_path

print(client_path())
```

`client_path()` returns the packaged `body-models-viser.js` bundle. A viser
integration can serve or inject this file into the browser, then call the
exported module API:

```ts
import { addBodyModel } from "./body-models-viser.js";

const handle = addBodyModel(scene, "/body", bodyModel);
handle.bodyPose = nextBodyPose;
```

The `bodyModel` object must provide the same small contract used by the
TypeScript tests:

- `faces`
- `skinWeights`
- `getRestPose()`
- `getBindParams(params)`
- `forward(params)`
- `forwardSkeleton(params)`

`addBodyModel()` returns a handle shaped like the existing
`body_models.extras.viser_plugin.add_body_model()` handle, including mutable
pose fields and `remove()`.

For development against the TypeScript source, import from `client/src/index.ts`
and run `npm test` from `client/`. Do not commit `client/dist/`; CI builds the
JavaScript bundle for releases.

## Commands

Generate Python references:

```sh
uv run --project ../body-models --no-sync scripts/generate_reference.py
```

Run Rust parity tests:

```sh
cargo test --release
```

Run the forward benchmark:

```sh
cargo run --release --bin bench_forward
```

Run one fixture through the Rust CLI:

```sh
cargo run --release -- \
  --model-data generated/model_data \
  --input fixtures/smpl/rest.json
```

Build and test the TypeScript package:

```sh
cd client
npm test
```

Open the side-by-side viser comparison:

```sh
uv run --project ../body-models --no-sync \
  scripts/compare_viser_plugins.py --model both --case shape_pose
```

## CI And Releases

GitHub Actions runs three checks:

- Rust formatting, clippy, and crate/bin tests.
- TypeScript build and tests from `client/src`.
- Python package build after generating the JavaScript bundle in CI.

The JavaScript bundle in `client/dist/` is not tracked by git. Releases build it
in GitHub Actions, package it into the Python wheel under
`body_models_viser/client/body-models-viser.js`, and publish with `uv publish`.
Configure `PYPI_USERNAME` and `PYPI_PASSWORD` repository secrets before creating
a GitHub release or running the publish workflow manually.

## Notes

The Rust implementation keeps the JSON format simple and builds sparse runtime
caches for the large sparse MHR matrices on first use. This keeps the generated
data easy to inspect while making repeated forwards fast enough for interactive
viewer work.
