from __future__ import annotations

import time

import body_models_viser as bmv
import numpy as np
import viser
from body_models.smpl.numpy import SMPL


POSE_JOINTS = [
    ("Spine1", 2),
    ("Spine2", 5),
    ("Spine3", 8),
    ("Neck", 11),
    ("L Shoulder", 15),
    ("R Shoulder", 16),
]


def main() -> None:
    server = viser.ViserServer()
    server.scene.add_grid(
        "/floor",
        width=4.0,
        height=4.0,
        plane="xz",
        cell_size=0.25,
        section_size=1.0,
        plane_opacity=0.02,
    )
    model = SMPL(gender="neutral")
    handle = bmv.add_body_model(server.scene, "/smpl", model, color=(173, 216, 230))
    add_controls(server, handle)
    server.scene.add_label("/labels/smpl", "SMPL", position=(0.0, 1.9, 0.0))

    while True:
        time.sleep(1.0 / 30.0)


def add_controls(server: viser.ViserServer, handle: bmv.SmplBodyHandle) -> None:
    with server.gui.add_folder("SMPL", expand_by_default=True):
        with server.gui.add_folder("Shape"):
            for i in range(10):
                add_slider(server, handle, f"beta{i}", "shape", (i,), -3.0, 3.0, 0.1)
        with server.gui.add_folder("Body Pose"):
            for label, joint in POSE_JOINTS:
                for axis, axis_name in enumerate("XYZ"):
                    add_slider(server, handle, f"{label} {axis_name}", "body_pose", (joint, axis), -1.5, 1.5, 0.05)


def add_slider(
    server: viser.ViserServer,
    handle: bmv.SmplBodyHandle,
    label: str,
    key: str,
    index: tuple[int, ...],
    lo: float,
    hi: float,
    step: float,
) -> None:
    gui = server.gui.add_slider(label, min=lo, max=hi, step=step, initial_value=0.0)

    @gui.on_update
    def _(_) -> None:
        params = np.asarray(handle.pose[key], dtype=np.float32).copy()
        params[index] = gui.value
        handle.set_pose(**{key: params})


if __name__ == "__main__":
    main()
