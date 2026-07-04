# 003 — Pin viser and declare the private-API surface

Written against commit `e3d4e18`. No dependencies; can land any time.

## Why

This package works by monkey-patching viser's private internals, but
`pyproject.toml` declares a completely unpinned dependency:

```toml
dependencies = [
  "body-models>=0.18.1",
  "jaxtyping",
  "nodeenv",
  "numpy",
  "viser",
]
```

Private API touched today (all from `body_models_viser/_body_model.py`):

| viser private surface | Used at |
|---|---|
| `scene._websock_interface` | `_runtime_state:335`, `add_body_model:327`, others |
| `viser._messages` module (`Message`, `RemoveSceneNodeMessage`, `RunJavascriptMessage`, `include_in_scene_serialization` kwarg) | imports, lines 25, 30-56 |
| `websock.get_message_serializer` (wrapped/replaced) | `_install_serializer_hook:353-366` |
| `websock._record_handles` | `_record_state_message:371` |
| `serializer._insert_message`, `serializer.as_html` (wrapped) | lines 361-392 |
| `websock._client_state_from_id`, `client_state.message_buffer.push/.flush` | lines 346, 396-401, 416-418 |
| `websock.register_handler`, `websock.on_client_connect`, `client.queue_message`, `client.get_message_buffer` | lines 327, 344-349, 404-410 |
| (TypeScript side) React fiber layout of the viser client, `viewer.mutable.current.messageQueue` / `.sendMessage` | `client/src/index.ts:60-67, 257-324` |

Any viser release can rename any of these and break the package **at runtime
for already-released versions on PyPI**. The current environment uses viser
1.0.29 and works.

## Steps

1. In `pyproject.toml`, pin a tested range:

   ```toml
   "viser>=1.0.29,<1.1",
   ```

   Rationale: patch releases within 1.0.x are unlikely to reshuffle internals;
   a minor bump gets consciously validated and the ceiling raised.

2. Run `uv lock` (updates `uv.lock`; this is the one permitted mutation).

3. Add a short section to `README.md` under "Development":

   ```markdown
   ## viser compatibility

   This package patches viser private internals (message serializer, websock
   client state, and the client React tree) to inject its runtime. The
   supported viser range is pinned in `pyproject.toml`; when raising the
   ceiling, run `uv run pytest` and `uv run scripts/visualize_models.py`
   against the new version and check a browser actually renders.
   ```

4. Verify: `uv run python -c "import body_models_viser"` succeeds;
   `uv run pytest` (if plan 001 landed) is green.

## Done criteria

- `pyproject.toml` pins `viser>=1.0.29,<1.1`; `uv.lock` consistent
  (`uv lock --check` passes).
- README documents the compatibility contract.
- No other files change.

## Boundaries

- Do not pin `numpy`/`jaxtyping`/`nodeenv` — they are used through stable
  public APIs.
- Do not add a CI job testing viser pre-releases; that is a maintainer
  decision noted below.

## Escape hatch

If the installed viser in `uv.lock` is already ≥1.1 and everything works,
pin `<1.2` around the working version instead — the point is a conscious
ceiling near a validated version, not the specific number.

## Maintenance note

When a new viser minor is released: raise the ceiling in a dedicated commit,
run the verification commands, and eyeball one browser session and one offline
HTML export (both code paths patch different internals). Plan 004 concentrates
all patched surfaces into one module (`_runtime.py`), which is the only place
to re-check on a viser upgrade.
