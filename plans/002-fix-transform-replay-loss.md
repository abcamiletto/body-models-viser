# 002 — Fix `set_transform` clobbering the stored pose for late clients

Written against commit `e3d4e18`. Depends on plan 001 (test harness).

## The bug

`body_models_viser/_body_model.py` keeps, per model, the last pose message in
`state.poses[name]` and replays it to late-joining browser clients and to
newly created offline-export serializers (`_replay_state`, line 421;
serializer hook, line 353).

`set_pose` / `set_identity` store a **full** `BodyModelsViserPoseMessage`
(rest vertices, skinning transforms, pose offsets). But `set_transform`
(lines 129-145) stores a **transform-only** message:

```python
def set_transform(self, **params):
    ...
    message = BodyModelsViserPoseMessage(
        name=self.name,
        rest_vertices=None,
        skinning_transforms=None,
        pose_offsets=None,
        global_rotation=np.ascontiguousarray(self.pose["global_rotation"], dtype="<f4"),
        global_translation=np.ascontiguousarray(self.pose["global_translation"], dtype="<f4"),
    )
    state = _runtime_state(self.scene)
    state.poses[self.name] = message          # <-- overwrites the full pose message
```

**Failure scenario:** call `handle.set_pose(body_pose=...)`, then
`handle.set_transform(global_translation=...)`, then open a new browser tab
(or export the scene to HTML). Replay sends the creation-time model message
(rest pose skinning) followed by the stored transform-only message. The new
client renders the **rest pose** at the new transform — the `set_pose` update
is silently lost. Clients already connected are unaffected, which is why this
went unnoticed.

The same applies with `global_rotation`, and after `set_identity` +
`set_transform`.

## The fix

Keep the wire message slim (live clients only need the changed transform), but
store a merged message so replay carries the full current state.
`BodyModelsViserPoseMessage` is a dataclass, so `dataclasses.replace` works:

```python
def set_transform(self, **params):
    invalid = params.keys() - {"global_rotation", "global_translation"}
    if invalid:
        raise ValueError(f"Invalid transform parameter(s): {', '.join(sorted(invalid))}.")
    self._update_pose(params)
    message = BodyModelsViserPoseMessage(
        name=self.name,
        rest_vertices=None,
        skinning_transforms=None,
        pose_offsets=None,
        global_rotation=np.ascontiguousarray(self.pose["global_rotation"], dtype="<f4"),
        global_translation=np.ascontiguousarray(self.pose["global_translation"], dtype="<f4"),
    )
    state = _runtime_state(self.scene)
    stored = state.poses.get(self.name)
    state.poses[self.name] = message if stored is None else dataclasses.replace(
        stored,
        global_rotation=message.global_rotation,
        global_translation=message.global_translation,
    )
    _record_state_message(self.scene, state, message)
    _queue_ready_clients(self.scene, message)
```

Notes for the executor:

- `dataclasses` is already imported in this file.
- When `stored is None` (no `set_pose`/`set_identity` yet), the transform-only
  message is correct as-is: replay sends the model message (which carries full
  creation-time skinning) followed by the transform update.
- `_record_state_message` / `_queue_ready_clients` must receive the **slim**
  `message`, not the merged one — live clients and in-progress recordings
  already have the earlier full pose message in their streams.

## Steps

1. Apply the change above to `set_transform` in
   `body_models_viser/_body_model.py`.
2. Add a regression test in `tests/test_body_model.py` (harness from plan
   001):
   - `add_body_model` → `set_pose(body_pose=nonzero)` →
     `set_transform(global_translation=[1, 2, 3])`.
   - Assert `state.poses["/stub"].rest_vertices is not None` and its
     `skinning_transforms` still reflect the posed values (using the stub's
     transforms-encode-pose trick from plan 001).
   - Assert `state.poses["/stub"].global_translation == [1, 2, 3]`.
   - Also test transform-before-pose: `add` → `set_transform(...)` only →
     stored message has `rest_vertices is None` (slim message is fine when no
     pose was stored).
3. Verify: `uv run pytest` — the new tests pass, all plan-001 tests still
   pass.

## Done criteria

- `uv run pytest` green.
- `git diff --stat` touches exactly `body_models_viser/_body_model.py` (one
  method) and `tests/test_body_model.py`.

## Boundaries

- Do not restructure anything else in `_body_model.py` — plan 004 does that.
- Do not change the message dataclasses or the TypeScript side; the wire
  format is untouched.

## Escape hatch

If plan 004 has already landed when you execute this, `set_transform` may live
in a different module with a `publish`-style helper — apply the same
store-merged/send-slim logic there; the regression test is unchanged.

## Maintenance note

Invariant to preserve in future changes: **`state.poses[name]`, when it has
been written by any full update, must always contain complete skinning data**
— it is the single source of truth for replay. Slim messages are a wire
optimization only.
