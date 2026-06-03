# body-models-viser

Browser-side body model evaluation for viser.

Python owns the body model asset loaders and the `prepare_identity()` /
`prepare_pose()` calls. TypeScript owns the browser model lifecycle and WASM
buffers. Rust owns the stateless `forward_vertices()` kernel.

## Usage

### Skinned Body Models

Use `add_body_model()` for non-rigid skinned body models such as SMPL, SMPL-X,
MANO, FLAME, MHR, SKEL, ANNY, GarmentMeasurements, and SOMA.

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
```

`add_body_model()` returns a model-specific handle with `set_pose(...)`,
`remove()`, `global_rotation`, `global_translation`, and the pose properties
supported by that model.

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

Use `add_rigid_body_model()` for rigid articulated models that expose
`link_names`, `link_mesh(link_name)`, and `forward_links(...)`.

```python
import body_models_viser as bmv

handle = bmv.add_rigid_body_model(server.scene, "/robot", model)
handle.set_pose(body_pose=handle.pose["body_pose"])
```

The rigid-body helper renders one static mesh per link, then updates link
transforms when the pose changes.

## Runtime

`bmv.add_body_model(scene, name, model)` does three things:

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
