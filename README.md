# body-models-viser

Browser-side SMPL evaluation for viser.

This package intentionally supports SMPL only right now. Python owns the
`body_models.smpl.numpy.SMPL` asset loader and shape-dependent
`prepare_identity()` step. The browser runtime stores the pose-dependent SMPL
state in WebAssembly and mirrors Python handle updates by calling Rust
`forward_vertices()`.

## Runtime

`bmv.add_body_model(scene, name, smpl)` does three things:

1. Injects `body-models-viser.js` and `body-models-viser.wasm`.
2. Sends SMPL weights, faces, material props, pose parameters, and the current
   identity as little-endian binary buffers.
3. Returns a `SmplBodyHandle`.

Every `handle.set_pose(...)` call sends pose parameters. If `shape` changes,
Python recomputes `prepare_identity()` and sends that identity again. TypeScript
only routes messages, copies binary buffers into WASM memory, and forwards the
resulting vertex buffer to viser as a regular mesh message.

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
