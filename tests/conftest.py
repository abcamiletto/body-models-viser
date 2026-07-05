from __future__ import annotations

import socket

import numpy as np
import pytest
import viser

import body_models_viser._body_model as bm


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
        # Encode the pose into the joint translations so tests can observe it.
        transforms = np.stack([np.eye(4, dtype=np.float32)] * 2)
        transforms[:, :3, 3] = pose["body_pose"]
        return {
            "skin_weights": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "faces": np.array([[0, 1, 0]], dtype=np.uint32),
            "rest_vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            "skinning_transforms": transforms,
            # No "pose_offsets" key: exercises the zeros default.
        }


class StubHandle(bm.BodyModelHandle):
    identity_keys = ("shape",)
    pose_keys = ("body_pose",)


class FakeBuffer:
    def __init__(self):
        self.messages = []

    def push(self, message):
        self.messages.append(message)

    def flush(self):
        pass


class FakeClientState:
    def __init__(self):
        self.message_buffer = FakeBuffer()


@pytest.fixture(scope="session")
def server():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = viser.ViserServer(port=port, verbose=False)
    yield server
    server.stop()


@pytest.fixture
def scene(server, monkeypatch):
    monkeypatch.setattr(bm, "_HANDLE_TYPES", {**bm._HANDLE_TYPES, StubModel: StubHandle})
    yield server.scene
    state = getattr(server.scene._websock_interface, "_body_models_viser", None)
    if state is not None:
        state.models.clear()
        state.poses.clear()
