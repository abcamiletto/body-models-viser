from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeClientState, StubHandle, StubModel

import body_models_viser as bmv
import body_models_viser._body_model as bm


def state_of(scene):
    return scene._websock_interface._body_models_viser


def test_add_records_model_message(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())

    assert isinstance(handle, StubHandle)
    assert set(handle.pose) == {"shape", "body_pose", "global_rotation", "global_translation"}
    assert all(value.dtype == np.float32 for value in handle.pose.values())

    message = state_of(scene).models["/stub"]
    assert message.vertex_count == 2
    assert message.faces.dtype == np.uint32
    assert message.rest_vertices.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(message.rest_vertices, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(message.pose_offsets, np.zeros((2, 3)))
    assert message.props["color"] == (180, 180, 180)
    assert "/stub" not in state_of(scene).poses


def test_unsupported_model_raises(scene):
    with pytest.raises(TypeError):
        bmv.add_body_model(scene, "/nope", object())


def test_set_pose_records_full_pose_message(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())
    posed = np.arange(6, dtype=np.float32).reshape(2, 3)

    handle.set_pose(body_pose=posed)

    message = state_of(scene).poses["/stub"]
    assert message.rest_vertices is not None
    np.testing.assert_array_equal(message.skinning_transforms[:, :3, 3], posed)
    np.testing.assert_array_equal(handle.body_pose, posed)


def test_set_identity_records_full_pose_message(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())

    handle.set_identity(shape=np.ones(3, dtype=np.float32))

    message = state_of(scene).poses["/stub"]
    assert message.rest_vertices is not None
    np.testing.assert_array_equal(handle.pose["shape"], np.ones(3))


def test_property_setters_route_to_updates(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())
    posed = np.ones((2, 3), dtype=np.float32)

    handle.body_pose = posed

    message = state_of(scene).poses["/stub"]
    np.testing.assert_array_equal(message.skinning_transforms[:, :3, 3], posed)


def test_invalid_keys_raise(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())
    with pytest.raises(ValueError):
        handle.set_pose(bogus=np.zeros(1))
    with pytest.raises(ValueError):
        handle.set_identity(bogus=np.zeros(1))
    with pytest.raises(ValueError):
        handle.set_transform(bogus=np.zeros(1))


def test_set_transform_before_any_pose_stores_slim_message(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())

    handle.set_transform(global_translation=np.array([1.0, 2.0, 3.0]))

    message = state_of(scene).poses["/stub"]
    assert message.rest_vertices is None
    np.testing.assert_array_equal(message.global_translation, [1.0, 2.0, 3.0])


def test_set_transform_after_pose_keeps_skinning_data(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())
    posed = np.arange(6, dtype=np.float32).reshape(2, 3)
    handle.set_pose(body_pose=posed)

    handle.set_transform(global_translation=np.array([1.0, 2.0, 3.0]))

    message = state_of(scene).poses["/stub"]
    assert message.rest_vertices is not None
    np.testing.assert_array_equal(message.skinning_transforms[:, :3, 3], posed)
    np.testing.assert_array_equal(message.global_translation, [1.0, 2.0, 3.0])


def test_replay_pushes_model_then_pose(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())
    handle.set_pose(body_pose=np.ones((2, 3), dtype=np.float32))

    client_state = FakeClientState()
    bm._replay_state(client_state, state_of(scene))

    types = [type(message).__name__ for message in client_state.message_buffer.messages]
    assert types == ["BodyModelsViserModelMessage", "BodyModelsViserPoseMessage"]


def test_remove_clears_state(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())
    handle.set_pose(body_pose=np.ones((2, 3), dtype=np.float32))

    handle.remove()

    assert "/stub" not in state_of(scene).models
    assert "/stub" not in state_of(scene).poses
    client_state = FakeClientState()
    bm._replay_state(client_state, state_of(scene))
    assert client_state.message_buffer.messages == []
    with pytest.raises(KeyError):
        handle.remove()
