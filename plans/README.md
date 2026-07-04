# Improvement Plans

Audit of `body-models-viser` at commit `e3d4e18` (2026-07-04). Goal: polish the
design, quarantine the unavoidable viser hacks behind clean seams, and make the
code minimal and maintainable. Full-file audit (every source file was read);
nothing was out of scope.

Selection note: written non-interactively — these are the top findings by
leverage (impact / effort, weighted by confidence). Lower-value items are
listed under "Considered and rejected / deferred" below.

## Execution order and dependencies

| # | Plan | Effort | Depends on | Status |
|---|------|--------|-----------|--------|
| 001 | [Characterization tests for the runtime plumbing](001-characterization-tests.md) | M | — | DONE |
| 002 | [Fix `set_transform` clobbering stored pose for late clients](002-fix-transform-replay-loss.md) | S | 001 | DONE |
| 003 | [Pin viser and declare the private-API surface](003-pin-viser-private-api.md) | S | — | DONE |
| 004 | [Quarantine viser transport hacks in `_runtime.py`](004-quarantine-viser-runtime.md) | M | 001, 002 | DONE |
| 005 | [Declarative per-model adapters, kill introspection](005-declarative-model-adapters.md) | M | 001, 004 | DONE |
| 006 | [Protocol naming, handle API naming, small cleanups](006-naming-and-cleanups.md) | M | 004, 005 | DONE |

All six executed on branch `polish-runtime` (2026-07-04). Verification:
Rust fmt/clippy/tests, TypeScript typecheck + build, 13 pytest tests, and
WASM/NumPy parity for the locally available models (ANNY, MHR, SOMA — the
others need licensed assets). End-to-end browser check: live client with
late-join replay after set_pose + set_transform, and an offline HTML export
(preload intercept → drain → restore) both render the posed body.

Execution notes discovered during plan 005 (they diverge from the plan text):
- SOMA, like ANNY, requires a zeroed `global_rotation` in `prepare_pose` —
  caught by the parity gate; it has the same override.
- MHR and SKEL introspected pose keys included `head_pose` even though the old
  handle classes had no such property; the declarative keys include it, so
  both handles gained the property.
- SMPLX's `prepare_pose` accepts `shape`/`expression` but ignores them
  (verified in body-models source); they are pose keys no longer.

001 must land first: it is the safety net every other plan's verification
depends on. 003 is independent and can land any time. 004 and 005 rewrite the
same file (`_body_model.py`) — do them in order, not in parallel.

## Verification commands (all plans)

```sh
cargo fmt --check && cargo clippy --release --lib -- -D warnings && cargo test --release --lib
cd client && npm ci && npm run typecheck && npm test   # npm test = full build
uv run pytest                                          # exists after plan 001
uv run scripts/check_model_parity.py                   # needs local model assets; run when available
```

## Considered and rejected / deferred

- **Rust kernel changes** (`src/skin.rs`, `src/lib.rs`): already minimal,
  tested, and clean. The dense O(V×J) weight loop is fine at SMPL scale. No
  action.
- **TypeScript runtime restructuring** (`client/src/index.ts`): the class is
  small and coherent. The React-fiber `findViewer()` and the message-queue
  patch cannot be removed without upstream viser support (see direction
  findings in plan 004's maintenance notes). Only naming changes in plan 006.
- **Replacing the `Array.prototype.push` preload patch**: no local alternative
  exists — offline-export HTML replays messages before the runtime installs.
  Contained and documented in plan 004 instead of removed.
- **`nodeenv` autobuild removal**: it is what makes `uv run` against a source
  checkout work with zero manual steps. Kept; minor cleanup in plan 006.
- **`viser.ScientificError`-style validation layers / defensive checks**:
  explicitly unwanted; the codebase should fail fast and loud.

## Upstream direction (for the maintainer, not for executors)

These erase hacks at the source instead of polishing around them; both are in
repos the maintainer controls or contributes to:

1. **body-models**: harmonize `GarmentMeasurements.prepare_pose` to per-part
   kwargs (like every other model) and make `ANNY.prepare_pose`'s
   `global_rotation` optional (like SMPL's `pelvis_rotation`). This deletes the
   `pack_pose` special case and the zeroed-`global_rotation` shim in this repo
   (plan 005 contains the local fallback if this doesn't happen).
2. **viser**: propose first-class extension points — custom binary message
   types and a supported way to access the client message queue. That would
   delete the serializer monkey-patches, the React-fiber traversal, and the
   `Array.prototype.push` preload patch. Until then, plan 004 quarantines them.
