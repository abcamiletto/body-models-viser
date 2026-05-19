# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "body-models",
#   "body-models-viser",
#   "numpy",
#   "viser",
# ]
# [tool.uv.sources]
# body-models-viser = { path = "..", editable = true }
# ///
"""Visualize supported body models with the body-models-viser plugin.

Model assets are resolved through the body-models config/cache.

Usage:
    uv run scripts/visualize_models.py
"""

from __future__ import annotations

import time
import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import body_models_viser as bmv
import numpy as np
import viser
from body_models.anny.numpy import ANNY
from body_models.base import BodyModel
from body_models.constants import Joint
from body_models.garment_measurements.numpy import GarmentMeasurements
from body_models.mhr.numpy import MHR
from body_models.smpl.numpy import SMPL
from body_models.smplh.numpy import SMPLH
from body_models.smplx.numpy import SMPLX
from body_models.soma.numpy import SOMA
from jaxtyping import Float

DISPLAY_GLOBAL_ROTATIONS = {
    "ANNY": (-np.pi / 2, 0.0, 0.0),
}

SMPL_POSE_JOINTS = [
    ("Spine1", 2),
    ("Spine2", 5),
    ("Spine3", 8),
    ("Neck", 11),
    ("L_Shoulder", 15),
    ("R_Shoulder", 16),
]

ANNY_PHENOTYPE_PARAMS = ["Gender", "Age", "Muscle", "Weight", "Height", "Proportions"]
ANNY_BODY_POSE_BONES = [
    ("Spine", 0),
    ("Spine1", 1),
    ("Spine2", 2),
    ("Neck", 3),
    ("L Shoulder", 7),
    ("L Arm", 8),
    ("R Shoulder", 12),
    ("R Arm", 13),
    ("L UpLeg", 17),
    ("R UpLeg", 22),
]

SOMA_BODY_POSE_JOINTS = [
    ("Spine1", 0),
    ("Spine2", 1),
    ("Chest", 2),
    ("Neck1", 3),
    ("L Shoulder", 5),
    ("L Arm", 6),
    ("L ForeArm", 7),
    ("R Shoulder", 9),
    ("R Arm", 10),
    ("R ForeArm", 11),
    ("L Leg", 13),
    ("R Leg", 18),
]
SOMA_HEAD_POSE_JOINTS = [("Head", 0)]
GARMENT_BODY_POSE_JOINTS = [
    ("Spine1", 2),
    ("Spine2", 3),
    ("Neck", 4),
    ("L Shoulder", 7),
    ("L Arm", 8),
    ("R Shoulder", 12),
    ("R Arm", 13),
    ("L Leg", 17),
    ("R Leg", 21),
]

GRID_COLS = 3
GRID_SPACING_X = 1.8
GRID_SPACING_Z = 1.8

MODEL_COLORS = {
    "SMPL": (173, 216, 230),
    "SMPLH": (216, 191, 216),
    "SMPLX": (255, 182, 193),
    "ANNY": (255, 218, 185),
    "Garment": (190, 220, 175),
    "MHR": (221, 160, 221),
    "SOMA": (250, 200, 200),
}

JOINT_MARKER_COLOR = (45, 120, 255)
JOINT_HIGHLIGHT_COLOR = (255, 210, 35)
JOINT_MARKER_RADIUS = 0.025
JOINT_HIGHLIGHT_RADIUS = 0.055
HAND_JOINT_NAMES = ("thumb", "index", "middle", "ring", "pinky")
CANONICAL_POSE_MODELS = ("SMPL", "SMPLH", "SMPLX", "ANNY", "Garment", "MHR", "SOMA")


@dataclass
class ModelState:
    model: BodyModel
    params: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ]
    color: tuple[int, int, int]
    display_global_rotation: Float[np.ndarray, "3"] | None = None
    hands: str = "default"
    body_handle: bmv.ViserBodyHandle | None = None
    changed: bool = True


@dataclass
class SliderHandle:
    handle: viser.GuiInputHandle
    initial: float
    key: str
    indices: tuple[int, ...]


@dataclass
class ModelControls:
    folder: viser.GuiFolderHandle
    sliders: list[SliderHandle]


def add_slider(
    server: viser.ViserServer,
    state: ModelState,
    label: str,
    *,
    lo: float,
    hi: float,
    step: float,
    initial: float,
    key: str,
    indices: tuple[int, ...],
) -> SliderHandle:
    handle = server.gui.add_slider(label, min=lo, max=hi, step=step, initial_value=initial)

    @handle.on_update
    def _(event: Any) -> None:
        state.params[key][indices] = event.target.value
        state.changed = True

    return SliderHandle(handle, initial, key, indices)


def betas(
    server: viser.ViserServer,
    state: ModelState,
    *,
    key: str,
    count: int,
    prefix: str = "beta",
    lo: float = -3.0,
    hi: float = 3.0,
    step: float = 0.1,
    initial: float = 0.0,
) -> list[SliderHandle]:
    return [
        add_slider(
            server,
            state,
            f"{prefix}{i}",
            lo=lo,
            hi=hi,
            step=step,
            initial=initial,
            key=key,
            indices=(i,),
        )
        for i in range(count)
    ]


def joint_xyz(
    server: viser.ViserServer,
    state: ModelState,
    *,
    key: str,
    joints: list[tuple[str, int]],
    lo: float = -1.5,
    hi: float = 1.5,
    step: float = 0.05,
    max_joints: int | None = None,
) -> list[SliderHandle]:
    handles = []
    for name, joint in joints:
        if max_joints is not None and joint >= max_joints:
            continue
        for axis, axis_name in enumerate("XYZ"):
            handles.append(
                add_slider(
                    server,
                    state,
                    f"{name} {axis_name}",
                    lo=lo,
                    hi=hi,
                    step=step,
                    initial=0.0,
                    key=key,
                    indices=(joint, axis),
                )
            )
    return handles


def reset_button(server: viser.ViserServer, handles: list[SliderHandle]) -> None:
    button = server.gui.add_button("Reset")

    @button.on_click
    def _(_: Any) -> None:
        for slider in handles:
            slider.handle.value = slider.initial


def mutable_params(
    params: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> dict[
    str,
    Float[np.ndarray, "dim"]
    | Float[np.ndarray, "items dim"]
    | Float[np.ndarray, "items rows cols"],
]:
    return {key: np.asarray(value).copy() for key, value in params.items()}


def apply_pose(state: ModelState, sliders: list[SliderHandle], pose_name: str) -> None:
    if pose_name == "tpose":
        preset = state.model.get_tpose(hands=state.hands) if state.model.has_hands else state.model.get_tpose()
    elif pose_name == "apose":
        preset = state.model.get_apose(hands=state.hands) if state.model.has_hands else state.model.get_apose()
    else:
        raise ValueError(f"Unknown pose: {pose_name}")

    updated_keys = set()
    for key in ("body_pose", "head_pose", "hand_pose", "global_rotation"):
        if key in preset and key in state.params:
            state.params[key] = np.asarray(preset[key]).copy()
            updated_keys.add(key)
    if state.display_global_rotation is not None:
        state.params["global_rotation"] = state.display_global_rotation.copy()
        updated_keys.add("global_rotation")
    for slider in sliders:
        if slider.key in updated_keys:
            slider.handle.value = float(state.params[slider.key][slider.indices])
    state.changed = True


def apply_hands(state: ModelState, sliders: list[SliderHandle], hands: Literal["default", "flat", "rest"]) -> None:
    preset = state.model.get_rest_pose(hands=hands)
    state.hands = hands
    state.params["hand_pose"] = np.asarray(preset["hand_pose"]).copy()
    for slider in sliders:
        if slider.key == "hand_pose":
            slider.handle.value = float(state.params[slider.key][slider.indices])
    state.changed = True


def set_gui_visible(handle: Any, visible: bool) -> None:
    handle.visible = visible
    if isinstance(handle, viser.GuiFolderHandle):
        for child in handle._children.values():
            set_gui_visible(child, visible)


def add_model_controls(server: viser.ViserServer, name: str, state: ModelState) -> ModelControls:
    handles: list[SliderHandle] = []
    with server.gui.add_folder(name, expand_by_default=True) as folder:
        if name in {"SMPL", "SMPLH", "SMPLX"}:
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=10)
            if name == "SMPLX":
                with server.gui.add_folder("Expression"):
                    handles += betas(server, state, key="expression", count=10, prefix="expr", lo=-2.0, hi=2.0)
            with server.gui.add_folder("Body Pose"):
                handles += joint_xyz(server, state, key="body_pose", joints=SMPL_POSE_JOINTS)

        elif name == "ANNY":
            with server.gui.add_folder("Phenotype"):
                for label in ANNY_PHENOTYPE_PARAMS:
                    handles.append(
                        add_slider(
                            server,
                            state,
                            label,
                            lo=0.0,
                            hi=1.0,
                            step=0.05,
                            initial=0.5,
                            key=label.lower(),
                            indices=(),
                        )
                    )
            with server.gui.add_folder("Pose"):
                handles += joint_xyz(
                    server,
                    state,
                    key="body_pose",
                    joints=ANNY_BODY_POSE_BONES,
                    max_joints=state.params["body_pose"].shape[0],
                )

        elif name == "MHR":
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=10)
            with server.gui.add_folder("Expression"):
                handles += betas(server, state, key="expression", count=15, prefix="expr", lo=-2.0, hi=2.0)

        elif name == "Garment":
            assert isinstance(state.model, GarmentMeasurements)
            with server.gui.add_folder("Shape"):
                handles += betas(server, state, key="shape", count=min(10, state.model.num_shape_components))
            with server.gui.add_folder("Body Pose"):
                handles += joint_xyz(
                    server,
                    state,
                    key="body_pose",
                    joints=GARMENT_BODY_POSE_JOINTS,
                    max_joints=state.params["body_pose"].shape[0],
                )

        elif name == "SOMA":
            assert isinstance(state.model, SOMA)
            identity_default = float(state.params["identity"][0])
            with server.gui.add_folder("Identity"):
                handles += betas(
                    server,
                    state,
                    key="identity",
                    count=min(10, state.model.identity_dim),
                    prefix="id",
                    lo=-1.0,
                    hi=1.0,
                    step=0.05,
                    initial=identity_default,
                )
            with server.gui.add_folder("Body Pose"):
                handles += joint_xyz(
                    server,
                    state,
                    key="body_pose",
                    joints=SOMA_BODY_POSE_JOINTS,
                    max_joints=state.params["body_pose"].shape[0],
                )
            with server.gui.add_folder("Head Pose"):
                handles += joint_xyz(
                    server,
                    state,
                    key="head_pose",
                    joints=SOMA_HEAD_POSE_JOINTS,
                    max_joints=state.params["head_pose"].shape[0],
                )

        else:
            raise ValueError(f"Unhandled model controls: {name}")

        reset_button(server, handles)
    return ModelControls(folder, handles)


def _joint_label(joint: Joint) -> str:
    return joint.value.replace("_", " ").title()


def _is_hand_joint(joint: Joint) -> bool:
    return any(name in joint.value for name in HAND_JOINT_NAMES)


def _is_left_joint(joint: Joint) -> bool:
    return joint.value.startswith("left_")


def standard_joints_tab(server: viser.ViserServer, tabs: Any, states: dict[str, ModelState]) -> Callable[[set[str]], None]:
    joint_indices = {
        name: {
            joint: state.model.joint_names.index(native_name)
            for joint, native_name in state.model.common_joints.items()
        }
        for name, state in states.items()
    }
    available_joints = [joint for joint in Joint if any(joint in indices for indices in joint_indices.values())]
    body_joints = [joint for joint in available_joints if not _is_hand_joint(joint)]
    hand_joints = [joint for joint in available_joints if _is_hand_joint(joint)]
    hand_sides = {
        "Left hand": [joint for joint in hand_joints if _is_left_joint(joint)],
        "Right hand": [joint for joint in hand_joints if not _is_left_joint(joint)],
    }
    markers = {}
    highlights = {}
    visible_joints: set[Joint] = set()
    selected_joint: Joint | None = None

    with tabs.add_tab("Joints", viser.Icon.HAND_CLICK):
        toggle_all = server.gui.add_button("Show all")
        checkboxes = {}
        with server.gui.add_folder("Body", expand_by_default=True):
            for joint in body_joints:
                checkboxes[joint] = server.gui.add_checkbox(_joint_label(joint), initial_value=False)
        with server.gui.add_folder("Hands", expand_by_default=True):
            for label in hand_sides:
                checkboxes[label] = server.gui.add_checkbox(label, initial_value=False)

    def select(joint: Joint | None) -> None:
        nonlocal selected_joint
        selected_joint = joint
        for key, marker in markers.items():
            marker.visible = key[1] in visible_joints
            highlights[key].visible = key[1] in visible_joints and key[1] == selected_joint

    for joint in body_joints:
        checkbox = checkboxes[joint]

        @checkbox.on_update
        def _(event: Any, checkbox_joint: Joint = joint) -> None:
            if event.target.value:
                visible_joints.add(checkbox_joint)
            else:
                visible_joints.discard(checkbox_joint)
            select(selected_joint)

    for label, joints in hand_sides.items():
        checkbox = checkboxes[label]

        @checkbox.on_update
        def _(event: Any, side_joints: list[Joint] = joints) -> None:
            if event.target.value:
                visible_joints.update(side_joints)
            else:
                visible_joints.difference_update(side_joints)
            select(selected_joint)

    @toggle_all.on_click
    def _(_: Any) -> None:
        show_all = len(visible_joints) < len(available_joints)
        toggle_all.label = "Hide all" if show_all else "Show all"
        for checkbox in checkboxes.values():
            checkbox.value = show_all

    for name, state in states.items():
        for joint in joint_indices[name]:
            key = (name, joint)
            marker = server.scene.add_icosphere(
                f"/joints/{name}/standard/{joint.value}",
                radius=JOINT_MARKER_RADIUS,
                color=JOINT_MARKER_COLOR,
                subdivisions=2,
                visible=False,
            )
            highlight = server.scene.add_icosphere(
                f"/joints/{name}/highlights/{joint.value}",
                radius=JOINT_HIGHLIGHT_RADIUS,
                color=JOINT_HIGHLIGHT_COLOR,
                subdivisions=2,
                opacity=0.75,
                visible=False,
            )

            @marker.on_click
            def _(_: Any, clicked_joint: Joint = joint) -> None:
                select(clicked_joint)

            @highlight.on_click
            def _(_: Any, clicked_joint: Joint = joint) -> None:
                select(clicked_joint)

            markers[key] = marker
            highlights[key] = highlight

    def update_markers(names: set[str]) -> None:
        for name in names:
            state = states[name]
            skeleton = state.model.forward_skeleton(**state.params)
            joint_positions = np.asarray(skeleton[:, :3, 3], dtype=np.float32)
            for joint, joint_index in joint_indices[name].items():
                markers[(name, joint)].position = joint_positions[joint_index]
                highlights[(name, joint)].position = joint_positions[joint_index]

    update_markers(set(states))
    return update_markers


def update_body_handle(server: viser.ViserServer, name: str, state: ModelState) -> None:
    if state.body_handle is None:
        state.body_handle = bmv.add_body_model(server.scene, f"/meshes/{name}", state.model, color=state.color)
    state.body_handle.set_pose(**state.params)


def load_models() -> dict[str, BodyModel]:
    model_specs: tuple[tuple[str, Callable[[], BodyModel]], ...] = (
        ("SMPL", lambda: SMPL(gender="neutral")),
        ("SMPLH", lambda: SMPLH(gender="neutral")),
        ("SMPLX", lambda: SMPLX(gender="neutral")),
        ("ANNY", ANNY),
        ("Garment", GarmentMeasurements),
        ("MHR", MHR),
        ("SOMA", lambda: SOMA(cache_identity=True, kernel="scipy")),
    )
    models = {}
    for name, make_model in model_specs:
        print(f"Loading {name}", flush=True)
        models[name] = make_model()
    return models


def init_states(models: dict[str, BodyModel]) -> dict[str, ModelState]:
    model_count = len(models)
    rows = (model_count + GRID_COLS - 1) // GRID_COLS
    states = {}
    for index, (name, model) in enumerate(models.items()):
        row, col = divmod(index, GRID_COLS)
        row_count = min(GRID_COLS, model_count - row * GRID_COLS)
        params = mutable_params(model.get_rest_pose())
        display_global_rotation = DISPLAY_GLOBAL_ROTATIONS.get(name)
        if display_global_rotation is not None:
            display_global_rotation = np.asarray(display_global_rotation, dtype=params["global_rotation"].dtype)
            params["global_rotation"] = display_global_rotation.copy()
        vertices = model.forward_vertices(**params)
        params["global_translation"] = np.asarray(
            (
                (col - 0.5 * (row_count - 1)) * GRID_SPACING_X,
                -float(vertices[..., 1].min()),
                (row - 0.5 * (rows - 1)) * GRID_SPACING_Z,
            ),
            dtype=params["global_translation"].dtype,
        )
        states[name] = ModelState(
            model=model,
            params=params,
            color=MODEL_COLORS[name],
            display_global_rotation=display_global_rotation,
        )
    return states


def add_labels(server: viser.ViserServer, states: dict[str, ModelState]) -> None:
    for name, state in states.items():
        vertices = state.model.forward_vertices(**state.params)
        position = np.asarray(state.params["global_translation"]).copy()
        position[1] = float(vertices[..., 1].max()) + 0.1
        server.scene.add_label(f"/labels/{name}", text=name, position=position)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", position=(0.0, 0.0, 0.0), plane="xz")
    server.gui.configure_theme(control_layout="fixed", control_width="large")

    models = load_models()
    print(f"Loaded {len(models)} models: {list(models.keys())}", flush=True)

    states = init_states(models)

    tabs = server.gui.add_tab_group()
    selected_model = next(iter(states))
    with tabs.add_tab("Models", viser.Icon.USER):
        model_dropdown = server.gui.add_dropdown(
            "Model",
            options=tuple(states.keys()),
            initial_value=selected_model,
        )
        controls = {name: add_model_controls(server, name, state) for name, state in states.items()}

    with tabs.add_tab("Poses"):
        with server.gui.add_folder("Body"):
            for label, pose_name in (("T-pose", "tpose"), ("A-pose", "apose")):
                button = server.gui.add_button(label)

                @button.on_click
                def _(_: Any, pose_name: str = pose_name) -> None:
                    for name in CANONICAL_POSE_MODELS:
                        apply_pose(states[name], controls[name].sliders, pose_name)

        with server.gui.add_folder("Hands"):
            for label, hands in (("Default hands", "default"), ("Flat hands", "flat"), ("Rest hands", "rest")):
                button = server.gui.add_button(label)

                @button.on_click
                def _(_: Any, hands: Literal["default", "flat", "rest"] = hands) -> None:
                    for name, state in states.items():
                        if state.model.has_hands:
                            apply_hands(state, controls[name].sliders, hands)

    def show_model_controls(name: str) -> None:
        for folder_name, model_controls in controls.items():
            set_gui_visible(model_controls.folder, folder_name == name)

    show_model_controls(selected_model)

    @model_dropdown.on_update
    def _(event: Any) -> None:
        show_model_controls(event.target.value)

    update_joint_markers = standard_joints_tab(server, tabs, states)
    add_labels(server, states)

    print("\nServer running", flush=True)
    while True:
        time.sleep(0.02)
        changed_models = set()
        for name, state in states.items():
            if state.changed:
                state.changed = False
                update_body_handle(server, name, state)
                changed_models.add(name)
        if changed_models:
            update_joint_markers(changed_models)


if __name__ == "__main__":
    main()
