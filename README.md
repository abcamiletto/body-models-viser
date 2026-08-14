# body-models-viser

Browser-side body model evaluation for viser.

Python owns the body model asset loaders and the `prepare_identity()` /
`prepare_pose()` calls. TypeScript owns the browser model lifecycle and WASM
buffers. Rust owns the stateless fallback kernels.

## Usage

### Skinned Body Models

Use `add_body_model()` for any `body_models.SkinnedModel`, including SMPL,
SMPL-X, MANO, FLAME, MHR, SOMA, SKEL, ANNY, and GarmentMeasurements.

```python
import body_models_viser as bmv
import viser
from body_models.smpl.numpy import SMPL

server = viser.ViserServer()
model = SMPL(gender="neutral")
handle = bmv.add_body_model(server.scene, "/smpl", model, color=(173, 216, 230))

pose = handle.body_pose.copy()
pose[2, 0] = 0.5
handle.body_pose = pose

translation = handle.global_translation.copy()
translation[1] = 0.25
handle.set_transform(global_translation=translation)

shape = handle.shape.copy()
shape[0] = 1.0
handle.set_identity(shape=shape)
```

`add_body_model()` returns a generic handle with `set_identity(...)`,
`set_pose(...)`, `set_transform(...)`, `remove()`, `global_rotation`,
`global_translation`, and the parameter properties declared by the model.

Pose correctives are disabled by default. Enable them explicitly when their
visual fidelity is needed:

```python
handle = bmv.add_body_model(
    server.scene,
    "/smpl",
    model,
    use_pose_correctives=True,
)
```

Correctives are evaluated in the client. Enabling them sends the model's dense
or sparse corrective basis once per shared asset, quantized to signed 16-bit
values, then sends only the small coefficient vector on each pose update. The
basis is not retransmitted per body or frame. Browser clients receive this
model data, so applications must ensure that doing so is compatible with the
asset's license.

### Skeletons

Use `add_skeleton()` for a standalone clickable skeleton. It takes joint
positions and a parent index for each joint.

```python
import body_models_viser as bmv

pose = model.get_rest_pose()
skeleton = model.forward_skeleton(**pose)
joint_positions = skeleton[:, :3, 3]

handle = bmv.add_skeleton(
    server.scene,
    "/skeleton",
    joint_positions,
    model.parents,
    joint_names=tuple(model.joint_names),
)

handle.visible = True
handle.joint_positions = joint_positions
```

## Runtime

`bmv.add_body_model(scene, name, model)` does three things:

1. Injects `body-models-viser.js` and `body-models-viser.wasm`.
2. Sends shared topology and sparse skin weights once, followed by the current
   identity and pose as little-endian binary buffers.
3. Returns a body model handle.

`handle.set_identity(...)` sends rest vertices and pose state.
`handle.set_pose(...)` sends only joint transforms and, when requested, pose
coefficients. `handle.set_transform(...)` sends only the global transform.
Without correctives, Rust applies sparse linear-blend skinning in WASM. With
correctives, a fused WebGPU kernel evaluates the corrective basis and skinning
together; the runtime falls back to WASM when WebGPU is unavailable. Both paths
preserve the basis representation exposed by `body-models`: dense bases remain
dense, while sparse bases remain sparse. The resulting vertex buffer is sent to
viser as a regular mesh message.

The browser protocol is model-agnostic. It consumes only the public
`SkinningSpec` and prepared identity and pose state from `body-models` 0.24.1
or newer.

## viser compatibility

This package patches viser private internals (message serializer, websock
client state, and the client React tree) to inject its runtime. The supported
viser range is pinned in `pyproject.toml`; when raising the ceiling, run
`uv run pytest` and `uv run scripts/visualize_models.py` against the new
version and check a browser actually renders.

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

Stress ten full-resolution SMPL-X bodies at 60 FPS:

```sh
OPENBLAS_NUM_THREADS=1 uv run scripts/stress_smplx.py
OPENBLAS_NUM_THREADS=1 uv run scripts/stress_smplx.py --use-pose-correctives
```

In the browser console, `BodyModelsViser.stats()` reports render FPS, model
update FPS, the corrective backend, and corrective batch time.
