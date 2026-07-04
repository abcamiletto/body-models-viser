# 005 — Declarative per-model adapters; kill signature introspection

Written against commit `e3d4e18`. Depends on plans 001 and 004 (execute after
the module split; file references below use post-004 layout — the adapter code
lives in `body_models_viser/_body_model.py`).

## Why

The routing of user parameters into `body-models` calls is currently done by
runtime signature introspection with hidden special cases
(`_body_model.py:502-534` at the audited commit):

```python
def _prepare_pose(model, params, identity):
    if "pose" in inspect.signature(model.prepare_pose).parameters:
        pose = pack_pose(np, params["pelvis_rotation"], params["body_pose"], params["head_pose"], params["hand_pose"])
        return model.prepare_pose(pose, identity=identity)

    pose_params = {}
    for key in _parameter_keys(model.prepare_pose, params.keys()):
        pose_params[key] = np.zeros_like(params[key]) if key == "global_rotation" else params[key]
    return model.prepare_pose(**pose_params, identity=identity)


def _pose_keys(model, keys):
    parameters = inspect.signature(model.prepare_pose).parameters
    if "pose" in parameters:
        return {"body_pose", "head_pose", "hand_pose", "pelvis_rotation"} & set(keys)
    return (set(parameters) & set(keys)) - {"global_rotation"}
```

Problems:

- `"pose" in signature` is a fingerprint for exactly one model
  (`GarmentMeasurements`, whose `prepare_pose` takes a packed array), but
  nothing says so; `pack_pose` is imported from
  `body_models.garment_measurements.pose` at module top as if it were generic.
- The `global_rotation → zeros` branch exists for exactly one model (ANNY,
  whose `prepare_pose` *requires* `global_rotation`; the browser applies the
  global transform in WASM instead, so Python must neutralize it). Also
  unstated.
- The same information is **already declared** a second time, correctly and
  per-model, by the handle classes (`_body_model.py:180-247`): every
  `_pose_property`/`_identity_property` line names exactly the keys that
  introspection re-derives. Two sources of truth, one of them reflective.

Verified facts about the upstream APIs (body-models 0.18.x, checked live):
`GarmentMeasurements.prepare_pose(self, pose, *, identity, ...)` is the only
packed-pose model; `ANNY.prepare_pose` is the only one with a required
`global_rotation`; all other models' `prepare_pose`/`prepare_identity` accept
exactly the kwargs named by their handle's declared properties. No listed
model is a subclass of another (the `isinstance`-chain ordering in
`_HANDLE_TYPES` with SMPL last is dead caution).

## Target design

One source of truth: the handle class declares its keys; properties are
generated from the declaration; the two upstream quirks become explicit,
named, per-model overrides.

```python
class BodyModelHandle:
    identity_keys: ClassVar[tuple[str, ...]] = ()
    pose_keys: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls) -> None:
        for key in cls.identity_keys:
            setattr(cls, key, _identity_property(key))
        for key in cls.pose_keys:
            setattr(cls, key, _pose_property(key))
    ...


class SmplBodyHandle(BodyModelHandle):
    identity_keys = ("shape",)
    pose_keys = ("body_pose", "pelvis_rotation")


class AnnyBodyHandle(BodyModelHandle):
    identity_keys = ("shape",)
    pose_keys = ("body_pose", "head_pose", "hand_pose")

    def _prepare_pose(self, identity):
        # ANNY.prepare_pose requires global_rotation; the browser applies the
        # global transform in WASM, so it must be zeroed here.
        return self.model.prepare_pose(
            **{key: self.pose[key] for key in self.pose_keys},
            global_rotation=np.zeros_like(self.pose["global_rotation"]),
            identity=identity,
        )


class GarmentMeasurementsBodyHandle(BodyModelHandle):
    identity_keys = ("shape",)
    pose_keys = ("body_pose", "head_pose", "hand_pose", "pelvis_rotation")

    def _prepare_pose(self, identity):
        # GarmentMeasurements.prepare_pose takes one packed pose array.
        pose = pack_pose(np, self.pose["pelvis_rotation"], self.pose["body_pose"],
                         self.pose["head_pose"], self.pose["hand_pose"])
        return self.model.prepare_pose(pose, identity=identity)
```

with the base implementations:

```python
def _prepare_identity(self) -> Any:
    return self.model.prepare_identity(**{key: self.pose[key] for key in self.identity_keys})

def _prepare_pose(self, identity: Any) -> Any:
    return self.model.prepare_pose(
        **{key: self.pose[key] for key in self.pose_keys}, identity=identity
    )
```

Consequences:

- Module-level `_prepare_pose`, `_prepare_identity`, `_parameter_keys`,
  `_pose_keys`, and the `inspect` import are deleted. Handle `__init__` no
  longer computes key sets (`_body_model.py:85-86`); `set_identity`/`set_pose`
  validate against the class attributes.
- Each handle class is now the complete, explicit contract for its model —
  keys and quirks in one visible place. Nine of eleven classes are two lines.
- `_HANDLE_TYPES` becomes a dict with exact-type lookup, killing the
  order-sensitivity:

  ```python
  _HANDLE_TYPES: dict[type, type[BodyModelHandle]] = {
      ANNY: AnnyBodyHandle,
      FLAME: FlameBodyHandle,
      ...
  }

  # in add_body_model:
  handle_type = _HANDLE_TYPES.get(type(model))
  if handle_type is None:
      raise TypeError(f"Unsupported body model {type(model).__name__}.")
  ```

  (Exact-type lookup means user subclasses of SMPL stop matching; that is an
  acceptable, loud behavior change — they were never tested.)

## Steps

1. Confirm the declared keys match the current property declarations for all
   ten handle classes at `_body_model.py:180-247` — transcribe, don't invent.
   The identity/pose split must be preserved exactly (e.g. FLAME's
   `expression` is an **identity** key; MHR's `expression` is an **identity**
   key; FLAME's `head_rotation` is a **pose** key).
2. Implement the base-class declaration + `__init_subclass__` + base
   `_prepare_identity`/`_prepare_pose`; convert all ten subclasses; add the
   two overrides (ANNY, GarmentMeasurements) with the comments shown above;
   move the `pack_pose` import into the GarmentMeasurements adapter region.
3. Convert `_HANDLE_TYPES` to a dict; simplify the dispatch in
   `add_body_model`.
4. Update call sites: wherever module-level `_prepare_identity(model, pose)` /
   `_prepare_pose(model, pose, identity)` were used (`add_body_model`,
   `set_identity`, `_publish_pose`), call the handle methods. Note
   `add_body_model` currently prepares identity/pose *before* constructing the
   handle — restructure so the handle is constructed first and then performs
   its initial prepare+publish, or keep a thin classmethod; executor's choice,
   but the prepare logic must live only on the handle.
5. Update `scripts/check_model_parity.py`: it imports the module-level
   `_prepare_identity` / `_prepare_pose` (lines 18, 38-39). Route it through
   the handle classes' methods instead (it needs no scene — construct the
   handle class without `add_body_model` via a small refactor, or instantiate
   `Handle.__new__`-free by giving the prepare methods no scene dependency;
   they only need `self.model` and `self.pose`, so constructing
   `handle_type(scene=None, name="x", model=model, pose=params, identity=None)`
   is acceptable for the script if `__init__` allows it — pick whichever keeps
   the script honest and short).
6. Update `tests/conftest.py` (plan 001): `StubHandle` now declares
   `identity_keys = ("shape",)`, `pose_keys = ("body_pose",)`; registration
   monkeypatch becomes a dict update. Behavioral assertions unchanged.
7. Verify: `uv run pytest`;
   `uv run scripts/check_model_parity.py` — **this is the critical gate**: it
   checks NumPy-vs-WASM vertex parity for all ten real models and will catch
   any mis-transcribed key or broken quirk handling. Run it if model assets
   are available locally; if not, STOP before merging and request a run from
   the maintainer.

## Done criteria

- `grep -n "inspect" body_models_viser/_body_model.py` returns nothing.
- `pack_pose` appears only inside the GarmentMeasurements handle's adapter.
- Every handle class states its keys as data; no runtime signature reflection
  anywhere in the package.
- `uv run pytest` green; parity script passes for all models (or an explicit
  maintainer sign-off is recorded in `plans/README.md` status).

## Boundaries

- **In scope:** `body_models_viser/_body_model.py`,
  `scripts/check_model_parity.py`, `tests/`.
- **Out of scope:** `_runtime.py` (the seam from plan 004 is unchanged), wire
  format, TypeScript, Rust, `body-models` itself.

## Escape hatches

- If any model's `prepare_identity`/`prepare_pose` rejects the transcribed
  kwargs (signature drift since body-models 0.18.x), STOP and report the
  model and signature — do not add introspection back to paper over it.
- If `add_body_model` restructuring (step 4) snowballs, it is acceptable to
  keep module-level thin wrappers that delegate to the handle methods — the
  hard requirement is single-source-of-truth keys and named per-model quirks,
  not a specific call graph.

## Maintenance note

Adding a new body model after this plan = one handle class declaring its keys
(+ an explicit `_prepare_pose` override if the upstream signature is
nonstandard) + one dict entry + one line in the parity script. If the
maintainer later harmonizes `GarmentMeasurements` / `ANNY` signatures upstream
in body-models (see `plans/README.md`, "Upstream direction"), the two
overrides simply get deleted.
