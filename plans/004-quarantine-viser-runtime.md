# 004 — Quarantine the viser transport hacks in `_runtime.py`

Written against commit `e3d4e18`. Depends on plans 001 (tests) and 002 (bug
fix — land it first so this refactor moves the *fixed* code).

## Why

`body_models_viser/_body_model.py` (535 lines) interleaves three unrelated
concerns:

1. **Transport hacks** — monkey-patching viser's serializer, injecting
   JavaScript, replaying state to late clients, the `Array.prototype.push`
   preload patch for offline exports. Fragile, viser-version-coupled, and the
   part that legitimately *must* be hacky until viser grows extension points.
2. **Model adaptation** — normalizing heterogeneous `body-models` APIs into
   one wire payload (`_prepare_pose`, `_prepare_identity`, `pack_pose`
   special-casing).
3. **Public API** — `add_body_model`, `BodyModelHandle` and its ten
   subclasses.

Because they share one file, every model-level change forces a reader through
transport hacks and vice versa. The fix: move concern 1 into a new
`body_models_viser/_runtime.py` with a three-function seam, so the hacks are
quarantined and the handle code reads like ordinary Python. This also removes
real duplication: two near-identical client-install paths, two copies of the
"store + record + broadcast" tail in `set_identity`/`set_pose`, and the
`connect_handler_installed` flag living apart from the rest of the state
setup.

## Target design

### `body_models_viser/_runtime.py` (new, ~200 lines)

Owns everything that touches viser internals. Contents, top-down:

```python
"""Transport layer: injects the JS/WASM runtime into viser and broadcasts
body-model messages.

Everything in this module touches viser PRIVATE internals (serializer,
websock client state, RunJavascriptMessage). It is the only module allowed
to. See README "viser compatibility".
"""

# --- message dataclasses (moved verbatim from _body_model.py:30-56) ---
class BodyModelsViserModelMessage(...)
class BodyModelsViserPoseMessage(...)
class BodyModelsViserReadyMessage(...)

# --- the seam used by _body_model.py ---
def get_state(scene) -> RuntimeState: ...
def broadcast(scene, message) -> None: ...

@dataclasses.dataclass
class RuntimeState:
    models: dict[str, BodyModelsViserModelMessage]
    poses: dict[str, BodyModelsViserPoseMessage]
    # private bookkeeping: ready_clients, installed_clients,
    # initialized_serializers

# --- below the fold: install/replay/serializer plumbing, JS snippets ---
```

Seam semantics:

- `get_state(scene)` — today's `_runtime_state` (`_body_model.py:334-350`),
  plus: it also installs the `on_client_connect` handler (today that lives in
  `add_body_model:324-329` behind the `connect_handler_installed` flag —
  moving it here deletes the flag) **and** an `on_client_disconnect` handler
  that discards the client id from `ready_clients` and `installed_clients`
  (today they grow forever; `viser.infra.WebsockServer` exposes
  `on_client_disconnect` — verified present in viser 1.0.29).
- `broadcast(scene, message)` — today's `_record_state_message` +
  `_queue_ready_clients` back to back (every current caller calls both, in
  that order). Takes any `viser._messages.Message`.
- Callers mutate `state.models` / `state.poses` directly, then `broadcast`.
  That keeps the seam at two functions instead of five specialized ones.

Consolidations while moving:

- **One client-install path.** `_install_connected_clients`
  (`_body_model.py:395-401`) and `_install_client_runtime` (404-410) do the
  same thing through different viser objects. Keep a single
  `_install_client(client_state)` operating on the `ClientState` from
  `websock._client_state_from_id`; the `on_client_connect` handler looks up
  the state by `client.client_id`. Both existing paths push
  `RunJavascriptMessage(_install_javascript())` and flush — preserve exactly
  that.
- `_queue_ready_clients` currently re-derives state via `_runtime_state(scene)`
  although every caller already has it; the merged `broadcast` uses the state
  it looks up once.
- Move `_install_javascript()` and `_preload_javascript()` here unchanged,
  including the `ensure_client_is_built()` call.
- Above the `_preload_javascript()` definition, add a short comment block
  stating the containment contract of the `Array.prototype.push` patch (it
  currently has none): why it exists (offline HTML replays messages before the
  runtime installs), its blast radius (page-global until `restore()`), and
  that `client/src/index.ts:drainPreload` is the only thing that restores it.

### `body_models_viser/_body_model.py` (shrinks to ~300 lines)

Keeps: `BodyModelHandle` + subclasses + property factories, `_HANDLE_TYPES`,
`add_body_model`, `_prepare_identity` / `_prepare_pose` / `_pose_message` /
`_parameter_keys` / `_pose_keys`. Imports the messages and the seam from
`._runtime`.

The duplicated update tail collapses. Today (`_body_model.py:104-127`), both
`set_identity` and `set_pose` end with six identical lines. After:

```python
def set_identity(self, **params):
    invalid = params.keys() - self.identity_keys
    if invalid:
        raise ValueError(f"Invalid identity parameter(s): {', '.join(sorted(invalid))}.")
    self._update_pose(params)
    self.identity = _prepare_identity(self.model, self.pose)
    self._publish_pose()

def set_pose(self, **params):
    invalid = params.keys() - self.pose_keys
    if invalid:
        raise ValueError(f"Invalid pose parameter(s): {', '.join(sorted(invalid))}.")
    self._update_pose(params)
    self._publish_pose()

def _publish_pose(self) -> None:
    prepared_pose = _prepare_pose(self.model, self.pose, self.identity)
    message = _pose_message(self.model, self.name, self.pose, self.identity, prepared_pose)
    state = _runtime.get_state(self.scene)
    state.poses[self.name] = message
    _runtime.broadcast(self.scene, message)
```

`set_transform` keeps its plan-002 merge logic, expressed via the same seam
(store merged, broadcast slim). `remove()` becomes: delete from
`state.models` / `state.poses`, `broadcast(RemoveSceneNodeMessage(name))`.

`add_body_model` also deduplicates against `_pose_message`: both build the
same skinning payload with the same `pose_offsets`-defaulting branch
(`_body_model.py:288-293` vs `485-490`). Extract one helper used by both:

```python
def _skinning_arrays(model, identity, prepared_pose):
    skinning = model.prepare_skinning(identity=identity, pose=prepared_pose)
    rest_vertices = skinning["rest_vertices"]
    pose_offsets = skinning.get("pose_offsets")
    if pose_offsets is None:
        pose_offsets = np.zeros_like(rest_vertices)
    return skinning, rest_vertices, pose_offsets
```

(Exact shape at the executor's discretion — the requirement is: the
`prepare_skinning` + `pose_offsets` default appears **once** in the package.)

### `body_models_viser/__init__.py`

Re-export unchanged names. Public API must not change in this plan.

## Steps

1. Land plans 001 and 002 first; run `uv run pytest` — green baseline.
2. Create `_runtime.py`; move the three message classes, `_RuntimeState`
   (rename `RuntimeState`), `_runtime_state` (rename `get_state`, absorb the
   connect-handler install, add disconnect pruning),
   `_install_serializer_hook`, `_record_state_message` +
   `_queue_ready_clients` (merge into `broadcast`),
   `_ensure_serializer_runtime`, `_install_serializer_preload`, the unified
   `_install_client`, `_replay_state`, `_install_javascript`,
   `_preload_javascript`. Order the file top-down: docstring, messages, seam,
   state, then plumbing.
3. Rewrite `_body_model.py` against the seam; delete
   `connect_handler_installed`, the duplicated tails, and the duplicated
   skinning-payload construction.
4. Update `tests/conftest.py` / tests for moved imports (mechanical only —
   behavioral assertions unchanged; the state object is now
   `_runtime.get_state(scene)` but still stashed on
   `scene._websock_interface._body_models_viser`, keep that attribute name so
   existing tests pass untouched if they read it directly).
5. Add one new test: connect-then-disconnect a fake client id via the
   registered handlers and assert `ready_clients` / `installed_clients` are
   pruned (call the handlers directly with a stub client object exposing
   `client_id`).
6. Verify: `uv run pytest`; `cd client && npm run typecheck && npm test`;
   `uv run scripts/check_model_parity.py` if model assets are available.
7. Manual smoke (if a browser is available):
   `uv run --no-sync scripts/visualize_models.py`, open the printed URL, move
   a slider, open a second tab, confirm the second tab shows the posed body.

## Done criteria

- `body_models_viser/_body_model.py` contains **zero** references to
  `_websock_interface` internals except via `_runtime`; grep gate:

  ```sh
  grep -n "_insert_message\|_record_handles\|get_message_serializer\|message_buffer\|_client_state_from_id" body_models_viser/_body_model.py
  ```

  returns nothing.
- `prepare_skinning(` appears exactly once in `body_models_viser/`.
- `uv run pytest` green; `import body_models_viser` exposes the same `__all__`.
- No wire-format change: message class names and field names identical
  (TypeScript untouched).

## Boundaries

- **In scope:** `body_models_viser/_body_model.py`, new
  `body_models_viser/_runtime.py`, `body_models_viser/__init__.py`, `tests/`.
- **Out of scope:** `client/`, `src/`, `_skeleton.py`, `_rigid_body.py`,
  `_client_autobuild.py`, scripts, wire format, the introspection-based
  `_prepare_pose`/`_prepare_identity` logic (plan 005), any renames of public
  attributes (plan 006).

## Escape hatches

- If merging `_record_state_message` + `_queue_ready_clients` into one
  `broadcast` changes observable ordering in a test, STOP and report — the
  current order (record first, then queue) must be preserved.
- If `on_client_disconnect` does not deliver a usable client id in viser
  1.0.29, drop the pruning (keep today's grow-forever behavior) and note it in
  the plan status rather than inventing a workaround.

## Maintenance note

After this plan, `_runtime.py` is the *only* file to audit on a viser upgrade
(plus `client/src/index.ts` on client-side changes), and the only file where
"hack" is an acceptable code-review answer. Anything added to `_body_model.py`
later that reaches into viser privates should be pushed down through the seam
instead. The long-term exit is upstream viser extension points (see
`plans/README.md`, "Upstream direction").
