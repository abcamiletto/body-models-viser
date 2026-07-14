"""Play ten full-resolution SMPL-X bodies for browser stress testing."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import viser
from body_models.smplx.numpy import SMPLX

import body_models_viser as bmv
from body_models_viser import _runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bodies", type=int, default=10)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 runs forever.")
    parser.add_argument("--use-pose-correctives", action="store_true")
    args = parser.parse_args()

    model = SMPLX(gender="neutral")
    server = viser.ViserServer(port=args.port)
    handles = [
        bmv.add_body_model(
            server.scene,
            f"/bodies/{index}",
            model,
            use_pose_correctives=args.use_pose_correctives,
            color=(110 + index * 9, 165, 220 - index * 8),
        )
        for index in range(args.bodies)
    ]
    columns = math.ceil(math.sqrt(args.bodies))
    for index, handle in enumerate(handles):
        row, column = divmod(index, columns)
        handle.set_transform(
            global_translation=np.array(
                [(column - (columns - 1) / 2) * 1.0, row * 1.25, 0.0],
                dtype=np.float32,
            )
        )

    initial_message = _runtime.get_state(server.scene).models[handles[0].name]
    pose_bytes = initial_message.skinning_transforms.nbytes
    if initial_message.pose_coefficients is not None:
        pose_bytes += initial_message.pose_coefficients.nbytes
    mode = "client correctives" if args.use_pose_correctives else "no correctives"
    print(
        f"{args.bodies} full-resolution SMPL-X bodies ({model.num_vertices:,} vertices), "
        f"{mode}; open http://localhost:{args.port}",
        flush=True,
    )
    print(
        f"Pose payload: {pose_bytes:,} bytes/body, "
        f"{pose_bytes * args.bodies * args.fps / 1e6:.2f} MB/s at {args.fps:g} FPS",
        flush=True,
    )

    rest_pose = model.get_rest_pose()
    base_pose = np.asarray(rest_pose["body_pose"], dtype=np.float32)
    period = 1.0 / args.fps
    started = report_started = time.perf_counter()
    frame = report_frames = 0
    total_update_time = 0.0
    try:
        while args.duration <= 0.0 or time.perf_counter() - started < args.duration:
            deadline = started + frame * period
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            update_started = time.perf_counter()
            phase = frame * period * 2.0 * math.pi
            for index, handle in enumerate(handles):
                pose = base_pose.copy()
                pose[0, 2] = 0.25 * math.sin(phase + index * 0.23)
                pose[1, 0] = 0.18 * math.cos(phase * 0.7 + index * 0.17)
                pose[2, 1] = 0.12 * math.sin(phase * 1.3 + index * 0.11)
                handle.set_pose(body_pose=pose)
            update_time = time.perf_counter() - update_started
            total_update_time += update_time
            frame += 1
            if time.perf_counter() - report_started >= 2.0:
                elapsed = time.perf_counter() - report_started
                rendered = frame - report_frames
                print(
                    f"server {rendered / elapsed:5.1f} FPS, "
                    f"{total_update_time / rendered * 1000:5.2f} ms/frame",
                    flush=True,
                )
                report_started = time.perf_counter()
                report_frames = frame
                total_update_time = 0.0
    finally:
        server.stop()


if __name__ == "__main__":
    main()
