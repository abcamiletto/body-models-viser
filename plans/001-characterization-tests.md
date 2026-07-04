# 001 — Characterization tests for the runtime plumbing

Written against commit `e3d4e18`. If `body_models_viser/_body_model.py` has
changed materially since (check `git log --oneline -- body_models_viser/`),
re-read it before starting and adapt line references.

## Why

`body_models_viser/_body_model.py` (535 lines) contains all the fragile logic
in this package: it monkey-patches viser internals to broadcast custom
messages, replays state to late-joining clients, and hooks scene serializers
for offline HTML exports. It has **zero tests**. The only tests in the repo
are one Rust unit test (`src/skin.rs`) and `scripts/check_model_parity.py`,
which needs downloaded body-model assets and never runs in CI.

Plans 002/004/005/006 all rewrite this file. This plan builds the safety net
first: a pytest suite that pins the observable behavior of the message/replay
machinery using a stub model, requiring no model assets and no browser.

## Repo context an executor needs

- Python ≥3.12 package managed with `uv`. Run everything as `uv run <cmd>`.
- The package under test is `body_models_viser/`; the module under test is
  `body_models_viser/_body_model.py`.
- `add_body_model(scene, name, model, ...)` is the entry point. It:
  1. dispatches on `isinstance(model, <type>)` over the `_HANDLE_TYPES` table
     (`_body_model.py:249-260`) to pick a handle class;
  2. computes skinning data via `model.prepare_identity` / `prepare_pose` /
     `prepare_skinning`;
  3. stores a `BodyModelsViserModelMessage` in a `_RuntimeState` stashed on
     `scene._websock_interface._body_models_viser`;
  4. returns a `BodyModelHandle` subclass.
- `handle.set_pose(...)` / `set_identity(...)` / `set_transform(...)` store a
  `BodyModelsViserPoseMessage` in `state.poses[name]` and broadcast it.
- Late-joining clients get state replayed via `_replay_state`
  (`_body_model.py:421-425`): the stored model message, then the stored pose
  message (if any).
- **Do not call anything that reaches `_install_javascript()`
  (`_body_model.py:428`) unless client assets are already built** — it
  triggers `ensure_client_is_built()`, which runs `npm ci` and a cargo wasm
  build. That means: avoid `websock.get_message_serializer(...)` and
  `_record_state_message` paths in tests *unless* the test is guarded (see
  step 5).

## Current state (excerpts to pin)

`_RuntimeState` and stashing (`_body_model.py:59-68, 334-350`):

```python
@dataclasses.dataclass
class _RuntimeState:
    ready_clients: set[int] = dataclasses.field(default_factory=set)
    installed_clients: set[int] = dataclasses.field(default_factory=set)
    ...
    models: dict[str, BodyModelsViserModelMessage] = dataclasses.field(default_factory=dict)
    poses: dict[str, BodyModelsViserPoseMessage] = dataclasses.field(default_factory=dict)
```

Handle update path (`_body_model.py:117-127`):

```python
def set_pose(self, **params):
    invalid = params.keys() - self.pose_keys
    if invalid:
        raise ValueError(...)
    self._update_pose(params)
    prepared_pose = _prepare_pose(self.model, self.pose, self.identity)
    message = _pose_message(self.model, self.name, self.pose, self.identity, prepared_pose)
    state = _runtime_state(self.scene)
    state.poses[self.name] = message
    ...
```

## Steps

### 1. Add a test dependency group

In `pyproject.toml`, extend the dev group:

```toml
[dependency-groups]
dev = [
    "pytest>=8",
    "wasmtime>=45.0.0",
]
```

Verify: `uv run pytest --version` prints a version.

### 2. Create the stub model — `tests/conftest.py`

Create `tests/` at repo root with a stub implementing the exact protocol the
adapter layer consumes. Keep the math trivially hand-checkable: 2 vertices,
2 joints, identity transforms.

```python
import numpy as np
import pytest


class StubModel:
    """Minimal body-models-protocol model: 2 vertices, 2 joints."""

    def get_rest_pose(self):
        return {
            "shape": np.zeros(3, dtype=np.float32),
            "body_pose": np.zeros((2, 3), dtype=np.float32),
            "global_rotation": np.zeros(3, dtype=np.float32),
            "global_translation": np.zeros(3, dtype=np.float32),
        }

    def prepare_identity(self, shape):
        return {"shape": np.asarray(shape)}

    def prepare_pose(self, body_pose, *, identity):
        return {"body_pose": np.asarray(body_pose)}

    def prepare_skinning(self, *, identity, pose):
        eye = np.eye(4, dtype=np.float32)
        # Encode pose into the transforms so tests can see pose changes:
        transforms = np.stack([eye, eye]).copy()
        transforms[:, :3, 3] = pose["body_pose"][:, :3]
        return {
            "skin_weights": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "faces": np.array([[0, 1, 0]], dtype=np.uint32),
            "rest_vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            "skinning_transforms": transforms,
            # NOTE: no "pose_offsets" key — pins the zeros-defaulting branch
            # at _body_model.py:290-293 and :487-490.
        }
```

Register a handle class for it and expose fixtures. `_HANDLE_TYPES` is a
module-level tuple today; extend it via monkeypatch so tests don't leak:

```python
import viser
import body_models_viser._body_model as bm


class StubHandle(bm.BodyModelHandle):
    shape = bm._identity_property("shape")
    body_pose = bm._pose_property("body_pose")


@pytest.fixture(scope="session")
def server():
    server = viser.ViserServer(port=0)  # if port=0 unsupported, pick a free port via socket
    yield server
    server.stop()


@pytest.fixture
def scene(server, monkeypatch):
    monkeypatch.setattr(bm, "_HANDLE_TYPES", bm._HANDLE_TYPES + ((StubModel, StubHandle),))
    return server.scene
```

If `viser.ViserServer(port=0)` errors, bind a free port first:
`socket.socket().bind(("", 0))` → `getsockname()[1]`, close, pass it in.

> **Plan 005 interaction:** plan 005 replaces `_HANDLE_TYPES` with a dict and
> may rename `_identity_property`. When executing plan 005, this conftest is
> expected to need a mechanical update — that is fine and called out there.

### 3. Test creation — `tests/test_body_model.py`

Pin what `add_body_model` records:

- Returns a `StubHandle`; `handle.pose` keys equal the rest-pose keys, arrays
  are float32 copies (mutating the input dict must not affect the handle).
- `state = scene._websock_interface._body_models_viser` exists after the call;
  `state.models["/stub"]` is a `BodyModelsViserModelMessage` with:
  - `vertex_count == 2`, `faces.dtype == np.uint32`,
  - `rest_vertices` equal to the stub's rest vertices, dtype `<f4`,
  - `pose_offsets` all zeros (the missing-key default),
  - `props["color"] == (180, 180, 180)` (defaults pass through).
- `state.poses` has no entry for `/stub`(creation stores only the model
  message; see `_body_model.py:317-319`).
- A second `add_body_model` with an unsupported model type raises `TypeError`.

### 4. Test updates and replay — same file

- `set_pose(body_pose=...)`: `state.poses["/stub"]` appears, its
  `skinning_transforms` reflect the new pose (translation column = posed
  values via the stub's encoding), `rest_vertices is not None`.
- `set_pose(bogus=...)` raises `ValueError`; `set_identity(bogus=...)` raises
  `ValueError`; `set_transform(bogus=...)` raises `ValueError`.
- Property sugar: `handle.body_pose = value` routes through `set_pose`
  (observable as a new stored pose message); `handle.shape = value` routes
  through `set_identity`.
- `set_transform(global_translation=...)`: stored message's
  `global_translation` updated. **Do not yet assert** `rest_vertices is not
  None` on the stored message after a prior `set_pose` — that is the bug plan
  002 fixes; its regression test lands there.
- Replay: after `add + set_pose`, call `bm._replay_state(FakeClientState(), state)`
  with a minimal fake:

  ```python
  class FakeBuffer:
      def __init__(self): self.messages = []
      def push(self, message): self.messages.append(message)

  class FakeClientState:
      def __init__(self): self.message_buffer = FakeBuffer()
  ```

  Assert order: model message first, then pose message; after `remove()`,
  replaying pushes nothing for that name.
- `remove()`: name gone from `state.models` and `state.poses`; a second
  `remove()` raises `KeyError` (current fail-fast behavior — pin it).

### 5. Serializer-hook test (guarded) — `tests/test_serializer.py`

This is the offline-export path. It calls `_install_javascript()`, so guard it:

```python
import pytest
from body_models_viser._client_autobuild import ASSETS

needs_client = pytest.mark.skipif(
    not all(asset.exists() for asset in ASSETS),
    reason="client bundle not built; run `cd client && npm test`",
)
```

Under the guard: after `add_body_model` + `set_pose`, call
`scene._websock_interface.get_message_serializer(lambda m: True)` and assert
the returned serializer received (via whatever list `_insert_message` appends
to — inspect `viser.infra.StateSerializer` to find it, likely a private list
of serialized messages): one `RunJavascriptMessage`, the model message, the
pose message, in that order. Assert the `as_html()` output contains
`__BODY_MODELS_VISER_PRELOAD__` in the `<head>`.

In CI this test runs in the `python-package` job only (client is built there —
see step 6); it self-skips elsewhere.

### 6. Wire pytest into CI

In `.github/workflows/ci.yml`, `python-package` job, after "Build client",
add:

```yaml
      - name: Test Python
        run: uv run pytest
```

Also add the same step (without the client build) as a new cheap job if
desired — not required; the guarded test skips when assets are missing.

## Done criteria

- `uv run pytest` passes locally with ≥ 12 assertions across the behaviors
  above; the serializer test runs (not skipped) when `client/dist` assets
  exist.
- `uv run pytest` passes with **no network access and no model assets**
  (except the initial `uv` dependency sync).
- CI config contains the new step; `git diff --stat` touches only `tests/`,
  `pyproject.toml`, `uv.lock`, `.github/workflows/ci.yml`.

## Boundaries

- **In scope:** `tests/` (new), `pyproject.toml` dev group, `uv.lock`,
  `.github/workflows/ci.yml`.
- **Out of scope:** any file in `body_models_viser/`, `client/`, `src/`. This
  plan changes zero production code. If a test reveals a bug, pin the
  *current* behavior with a comment naming the follow-up plan (002 already
  covers the known transform-replay bug); do not fix it here.

## Escape hatches

- If `viser.ViserServer` cannot start headless in CI (port/sandbox issues),
  STOP trying to run a real server and instead build a fake
  `scene._websock_interface` exposing `register_handler`,
  `get_message_serializer`, `_client_state_from_id`, and `on_client_connect`
  — the runtime code only touches those. Report which path was taken.
- If `StateSerializer`'s internals make asserting inserted messages
  unreasonable, assert on `as_html()` content only and note the gap.

## Maintenance note

These are characterization tests: when plans 004/005/006 intentionally change
internals (module paths, `_HANDLE_TYPES` shape), update imports/registration
mechanically but keep the behavioral assertions identical. Any *behavioral*
assertion change outside plan 002's fix is a red flag in review.
