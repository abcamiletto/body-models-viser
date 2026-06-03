# body-models-viser

Browser-side body model evaluation for viser.

Python owns the body model asset loaders and the `prepare_identity()` /
`prepare_pose()` calls. TypeScript owns the browser model lifecycle and WASM
buffers. Rust owns the stateless `forward_vertices()` kernel.

## Runtime

`bmv.add_body_model(scene, name, smpl)` does three things:

1. Injects `body-models-viser.js` and `body-models-viser.wasm`.
2. Sends faces, material props, current identity, and current pose as
   little-endian binary buffers.
3. Returns a body model handle.

Every `handle.set_pose(...)` call sends prepared pose data. If identity-changing
parameters such as `shape`, `expression`, or `scale_params` change, Python
recomputes `prepare_identity()` and sends that identity again. TypeScript copies
changed buffers into persistent WASM memory, calls the Rust
`forward_vertices()` kernel, and forwards the resulting vertex buffer to viser
as a regular mesh message.

The browser protocol is model-agnostic: Python adapts each model to a shared
runtime payload containing skinning weights, rest vertices, skinning transforms,
and pose offsets. Adding another body model should only require a Python
preparation adapter when its `body-models` outputs need normalization.

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

Check NumPy/WASM vertex parity for all supported models:

```sh
uv run scripts/check_model_parity.py
```
