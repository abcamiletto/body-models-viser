# body-models-viser

Browser-side body model evaluation for viser.

Python owns the body model asset loaders and the `prepare_identity()` /
`prepare_pose()` calls. TypeScript owns the browser model lifecycle and WASM
buffers. Rust owns the stateless fallback kernels.

## Usage

### Skinned Body Models

Use `add_body_model()` for non-rigid skinned body models such as SMPL, SMPL-X,
MANO, FLAME, SKEL, ANNY, and GarmentMeasurements.

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

Correctives are always evaluated in the client. Enabling them sends a 16-bit
quantized corrective basis once per shared model asset, then sends only the
small pose-coefficient vector on each update. For full-resolution SMPL-X the
one-time basis is about 29.1 MiB (30.6 MB); it is not retransmitted per body or
frame. The quantized model data is available to browser clients, so
applications must ensure that this is compatible with the model asset's
license.

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

### Rigid Body Models

Use `add_rigid_body_model()` for any `body_models.RigidBodyModel`, such as G1,
BrainCo, SmplHumanoid, and MyoFullBody.

```python
import body_models_viser as bmv
from body_models import create_model

model = create_model("g1")
handle = bmv.add_rigid_body_model(server.scene, "/robot", model)
handle.set_pose(body_pose=handle.pose["body_pose"])
```

The rigid-body helper bakes one static link-local mesh per link from the rest
pose (`forward_meshes()` sliced with the model's link metadata), then only
updates link transforms from `forward_links()` when the pose changes.

## Runtime

`bmv.add_body_model(scene, name, model)` does three things:

1. Injects `body-models-viser.js` and `body-models-viser.wasm`.
2. Sends shared topology and sparse skin weights once, followed by the current
   identity and pose as little-endian binary buffers.
3. Returns a body model handle.

`handle.set_identity(...)` sends rest vertices and pose state.
`handle.set_pose(...)` sends only joint transforms and, when requested, pose
coefficients. `handle.set_transform(...)` sends only the global transform.
Models with an explicit `posedirs` basis use `prepare_pose(skip_vertices=True)`,
so Python never evaluates their per-vertex pose-corrective offsets.

Without correctives, Rust applies sparse linear-blend skinning in WASM. With
correctives, a fused WebGPU kernel evaluates correctives and skinning together;
the runtime falls back to WASM when WebGPU is unavailable. In both cases the
resulting vertex buffer is forwarded to viser as a regular mesh message.

The browser protocol remains model-agnostic. Client correctives require the
model to use compatible `posedirs`, `parents`, and prepared
`skeleton_transforms`, as the SMPL, SMPL-H, SMPL-X, MANO, and FLAME
implementations in `body-models` do. SKEL uses a different corrective feature
mapping and therefore currently renders without correctives. Models such as
MHR and SOMA currently expose only server-computed pose offsets; they are
rejected instead of silently doing per-vertex server work or rendering without
their required deformation.

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
