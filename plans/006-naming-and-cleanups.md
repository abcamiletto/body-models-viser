# 006 — Protocol naming, handle API naming, small cleanups

Written against commit `e3d4e18`. Depends on plans 004 and 005 (this plan
renames things those plans move; do it last). Backwards compatibility is
explicitly **not** a goal — the maintainer has opted into breaking changes to
keep complexity low. Public-facing renames must be reflected in `README.md`.

This plan is a set of independent, mechanical cleanups. Each section has its
own verification; they can be committed separately.

## A. One name for skin weights across the wire

The same array is called `skin_weights` by body-models
(`prepare_skinning()["skin_weights"]`), `lbs_weights` in the Python message
and TypeScript (`_body_model.py:34,308`, `client/src/index.ts:17,142`), and
`lbs_weights` in Rust (`src/lib.rs:24-25`). Pick the body-models name —
`skin_weights` — end to end:

1. `body_models_viser/_runtime.py`: message field `lbs_weights` →
   `skin_weights`.
2. `body_models_viser/_body_model.py`: construction site keyword.
3. `client/src/index.ts`: `ModelMessage.lbs_weights` → `skin_weights`;
   `MeshState.lbsWeights` → `skinWeights`; all uses.
4. `src/lib.rs` / `src/skin.rs`: `lbs_weights*` parameter/field names →
   `skin_weights*` (no ABI change — names only).

Note: the Python field name **is** the wire name (viser serializes dataclass
fields), so Python and TypeScript must change in the same commit. Rust is
positional and independent.

Verify: `grep -rn "lbs" body_models_viser client/src src` returns nothing;
`cargo test --release --lib`; `cd client && npm run typecheck && npm test`;
`uv run pytest`.

## B. Drop the redundant `vertex_count` wire field

`BodyModelsViserModelMessage.vertex_count` (`_body_model.py:33,307`) is always
`len(rest_vertices) / 3`, and TypeScript already receives `rest_vertices` in
the same message. Remove the field; in `client/src/index.ts:148` replace

```ts
outputVertices: this.allocF32(message.vertex_count * 3),
```

with

```ts
outputVertices: this.allocF32(message.rest_vertices.length),
```

Verify: typecheck + build + pytest as in section A; grep for `vertex_count`
returns nothing.

## C. Rename `BodyModelHandle.pose` → `params`; make the prepared identity private

`handle.pose` (`_body_model.py:83`) actually holds **all** parameters —
`shape` (identity), pose parts, and the global transform — because it is
initialized from `model.get_rest_pose()`. Meanwhile `handle.identity` holds
not identity *parameters* but the cached output of `prepare_identity()`.
Both names mislead.

1. Rename attribute `pose` → `params` throughout the handle code (including
   `_update_pose` → `_update_params`).
2. Rename attribute `identity` → `_prepared_identity` (it is a cache, not
   API).
3. Update `scripts/visualize_models.py:66` (`handle.pose[key]` →
   `handle.params[key]`).
4. README: the usage example reads `handle.body_pose.copy()` etc. and is
   unaffected, but scan for `\.pose` references and fix any.

Verify: `grep -rn "\.pose\b" body_models_viser scripts README.md` shows only
rigid-body/skeleton legitimate uses; `uv run pytest`;
`uv run --no-sync scripts/visualize_models.py` starts without error (Ctrl-C
after the URL prints).

## D. `_rigid_body.py`: collapse the copy-pasted assert-properties

`body_models_viser/_rigid_body.py:62-103` has four identical property pairs
guarded by `assert`, e.g.:

```python
@property
def body_pose(self):
    assert "body_pose" in self.pose, f"{self.model_name} does not support 'body_pose'."
    return self.pose["body_pose"]
```

Replace all four pairs with the same factory idiom the body-model handles use:

```python
def _pose_property(key: str) -> property:
    def get(self: ViserRigidBodyModelHandle) -> np.ndarray:
        return self.pose[key]

    def set(self: ViserRigidBodyModelHandle, value: np.ndarray) -> None:
        self.set_pose(**{key: value})

    return property(get, set)


class ViserRigidBodyModelHandle:
    body_pose = _pose_property("body_pose")
    hand_pose = _pose_property("hand_pose")
    global_rotation = _pose_property("global_rotation")
    global_translation = _pose_property("global_translation")
```

Unsupported keys then fail loudly as `KeyError` from `self.pose[key]` on read,
and `set_pose` validation on write. In `set_pose` (lines 105-116):

- replace `assert name in self.pose, ...` with
  `raise ValueError(f"{self.model_name} does not support {name!r}.")` on
  invalid keys (match the body-model handles' error style);
- delete the `np.array_equal` change-detection short-circuit — it is
  premature optimization that costs a full array compare on every call and
  silently skips `_apply_pose`; always copy and apply.

Verify: `uv run pytest` (add a small test only if a rigid model stub is
trivial; otherwise `uv run python -c "import body_models_viser"` suffices —
this module has no test coverage and gaining it is out of scope here).

## E. `_client_autobuild.py`: explicit checks, deduplicated npm path

`body_models_viser/_client_autobuild.py`:

1. Lines 30-33 use `asset.stat()` purely for its `FileNotFoundError` side
   effect. Replace with an explicit, loud message:

   ```python
   if not CLIENT_DIR.exists():
       for asset in ASSETS:
           if not asset.exists():
               raise FileNotFoundError(
                   f"Prebuilt client asset missing: {asset}. "
                   "The package was built without client assets."
               )
       return
   ```

2. The `npm` path construction (bin dir + `.cmd` suffix on Windows) is
   duplicated at lines 46-49 and 60-62. Extract `_npm_path(env_dir) ->
   pathlib.Path` used by both.

Verify: `uv run python -c "from body_models_viser._client_autobuild import ensure_client_is_built; ensure_client_is_built()"`
succeeds (client dir exists in a checkout).

## F. Stop pretending the private manifests are versioned

Three version numbers exist: `pyproject.toml` 0.3.3 (the real one, drives
releases via `.github/workflows/release.yml`), `client/package.json` 0.3.3
(private, never published), `Cargo.toml` 0.3.0 (already drifted, never
published). Set both non-Python manifests to `0.0.0` so nobody mistakes them
for release state, and only `pyproject.toml` gets bumped on release.

Also update `body_models_viser/__init__.py`'s `__version__` — it duplicates
`pyproject.toml` by hand (both say 0.3.3 today). Replace with:

```python
from importlib.metadata import version

__version__ = version("body-models-viser")
```

Verify: `uv run python -c "import body_models_viser; print(body_models_viser.__version__)"`
prints `0.3.3`; `cargo build --release` fine; `cd client && npm test` fine.

## Done criteria

- All section-level greps/commands pass.
- Full gates: `cargo fmt --check && cargo clippy --release --lib -- -D warnings
  && cargo test --release --lib`; `cd client && npm ci && npm run typecheck &&
  npm test`; `uv run pytest`; parity script if assets available.
- README consistent with the renamed API.

## Boundaries

- **Out of scope:** `_skeleton.py` (its shape asserts are cheap and its code
  is already clean — reviewed and deliberately left alone), the release
  workflow, `_runtime.py` structure, any behavior changes beyond D's
  short-circuit removal.

## Escape hatches

- Section A/B change the wire format between Python and the bundled JS. They
  are safe only because the JS ships inside the same wheel — but **offline
  HTML exports recorded with an older version replay old field names**. If the
  maintainer needs old exports to keep working with a new bundle, STOP on A/B
  and confirm first (default assumption: exports embed their own JS snapshot,
  so they are self-contained and unaffected — verify by checking that exported
  HTML inlines the `RunJavascriptMessage` with the runtime source).
- If `importlib.metadata.version` fails in the dev checkout (editable install
  edge case), keep the literal `__version__` and note it; do not add fallback
  try/except.

## Maintenance note

After F, a release is: bump `pyproject.toml`, merge to master — the workflow
detects the change, builds, publishes, tags. No other file mentions the
version.
