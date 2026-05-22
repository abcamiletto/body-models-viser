# body-models-viser

Browser-side body model evaluation for viser.

This package intentionally supports SMPL only right now. Python owns the
`body_models.smpl.numpy.SMPL` asset loader and the `prepare_identity()` /
`prepare_pose()` calls. TypeScript owns the browser model lifecycle and WASM
buffers. Rust owns stateless `forward_vertices()` kernels.

## Runtime

`bmv.add_body_model(scene, name, smpl)` does three things:

1. Injects `body-models-viser.js` and `body-models-viser.wasm`.
2. Sends faces, material props, current identity, and current pose as
   little-endian binary buffers.
3. Returns a `SmplBodyHandle`.

Every `handle.set_pose(...)` call sends prepared pose data. If `shape` changes,
Python recomputes `prepare_identity()` and sends that identity again. TypeScript
copies changed buffers into persistent WASM memory, calls the Rust
`forward_vertices()` kernel, and forwards the resulting vertex buffer to viser
as a regular mesh message.

The browser protocol is model-agnostic: model messages carry a `model_type`
discriminator, and pose/remove/replay lifecycle is shared. Adding another body
model should only add the Python preparation adapter and one Rust
`forward_vertices()` kernel for that model.

## Development

Build the Rust crate:

```sh
cargo test
```

Build the browser bundle and WASM:

```sh
cd client
npm test
```

The browser build requires a Rust toolchain with `wasm32-unknown-unknown`
installed, for example:

```sh
rustup target add wasm32-unknown-unknown
```

Run the small visualizer:

```sh
uv run --no-sync scripts/visualize_models.py
```

Check SMPL NumPy/WASM vertex parity:

```sh
uv run scripts/check_smpl_parity.py
```
