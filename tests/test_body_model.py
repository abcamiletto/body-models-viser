from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeClientState, StubModel

import body_models
import body_models_viser as bmv
from body_models_viser import _runtime


class CorrectiveStubModel(StubModel):
    parents = [-1, 0]

    def __init__(self) -> None:
        self.posedirs = np.zeros((9, 6), dtype=np.float32)
        self.posedirs[0, 0] = 0.25
        self.server_corrective_evaluations = 0

    def prepare_pose(self, body_pose, *, identity, skip_vertices=False):
        angle = float(body_pose[1, 2])
        c, s = np.cos(angle), np.sin(angle)
        transforms = np.stack([np.eye(4, dtype=np.float32)] * 2)
        transforms[1, :3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        pose = {
            "skeleton_transforms": transforms,
            "skinning_transforms": transforms,
        }
        if not skip_vertices:
            self.server_corrective_evaluations += 1
            coefficients = (transforms[1, :3, :3] - np.eye(3)).reshape(-1)
            pose["pose_offsets"] = (coefficients @ self.posedirs).reshape(2, 3)
        return pose


body_models.SkinnedModel.register(CorrectiveStubModel)


class ServerOffsetStubModel(StubModel):
    def prepare_skinning(self, *, identity, pose):
        skinning = super().prepare_skinning(identity=identity, pose=pose)
        skinning["pose_offsets"] = np.ones_like(skinning["rest_vertices"])
        return skinning


body_models.SkinnedModel.register(ServerOffsetStubModel)


def state_of(scene):
    return scene._websock_interface._body_models_viser


def test_add_records_shared_asset_and_model(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())

    assert isinstance(handle, bmv.BodyModelHandle)
    assert handle.use_pose_correctives is False
    assert set(handle.params) == {"shape", "body_pose", "global_rotation", "global_translation"}
    assert all(value.dtype == np.float32 for value in handle.params.values())

    state = state_of(scene)
    message = state.models["/stub"]
    asset = next(iter(state.assets.values())).message
    assert asset.faces.dtype == np.uint32
    assert asset.corrective_basis is None
    assert asset.corrective_scales is None
    np.testing.assert_array_equal(message.rest_vertices, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(asset.skin_weight_offsets, [0, 1, 2])
    np.testing.assert_array_equal(asset.skin_weight_indices, [0, 1])
    np.testing.assert_array_equal(asset.skin_weight_values, [1.0, 1.0])
    assert message.props["color"] == (180, 180, 180)


def test_assets_are_shared_between_instances(scene):
    model = StubModel()
    first = bmv.add_body_model(scene, "/first", model)
    second = bmv.add_body_model(scene, "/second", model)

    state = state_of(scene)
    assert len(state.assets) == 1
    asset = next(iter(state.assets.values()))
    assert state.models[first.name].asset_id == state.models[second.name].asset_id
    assert asset.refcount == 2


def test_duplicate_name_is_rejected(scene):
    bmv.add_body_model(scene, "/stub", StubModel())

    with pytest.raises(ValueError, match="already exists"):
        bmv.add_body_model(scene, "/stub", StubModel())


def test_pose_correctives_are_only_evaluated_in_client(scene):
    model = CorrectiveStubModel()
    handle = bmv.add_body_model(
        scene,
        "/corrective",
        model,
        use_pose_correctives=True,
    )

    state = state_of(scene)
    asset = next(iter(state.assets.values())).message
    assert asset.corrective_basis is not None
    assert asset.corrective_basis.dtype == np.int16
    assert asset.corrective_basis.shape == (6, 9)
    assert asset.corrective_scales is not None
    assert state.models["/corrective"].pose_coefficients is not None
    assert model.server_corrective_evaluations == 0

    pose = np.zeros((2, 3), dtype=np.float32)
    pose[1, 2] = 0.5
    handle.set_pose(body_pose=pose)

    coefficients = state.models["/corrective"].pose_coefficients
    assert coefficients is not None
    assert np.any(coefficients != 0.0)
    assert model.server_corrective_evaluations == 0


def test_correctives_require_a_basis(scene):
    with pytest.raises(ValueError, match="does not expose compatible client-side pose correctives"):
        bmv.add_body_model(scene, "/stub", StubModel(), use_pose_correctives=True)


def test_server_only_pose_offsets_are_rejected(scene):
    with pytest.raises(ValueError, match="only exposes server-side pose offsets"):
        bmv.add_body_model(scene, "/offsets", ServerOffsetStubModel())


def test_unsupported_model_raises(scene):
    with pytest.raises(TypeError):
        bmv.add_body_model(scene, "/nope", object())


def test_set_pose_records_pose_only_message(scene, monkeypatch):
    messages = []
    monkeypatch.setattr(_runtime, "broadcast", lambda scene, message: messages.append(message))
    handle = bmv.add_body_model(scene, "/stub", StubModel())
    posed = np.arange(6, dtype=np.float32).reshape(2, 3)

    handle.set_pose(body_pose=posed)

    message = state_of(scene).models["/stub"]
    np.testing.assert_array_equal(message.skinning_transforms[:, :3, 3], posed)
    np.testing.assert_array_equal(handle.body_pose, posed)
    broadcast = messages[-1]
    assert isinstance(broadcast, _runtime.BodyModelsViserPoseMessage)
    assert not hasattr(broadcast, "rest_vertices")


def test_set_identity_uses_identity_message(scene, monkeypatch):
    messages = []
    monkeypatch.setattr(_runtime, "broadcast", lambda scene, message: messages.append(message))
    handle = bmv.add_body_model(scene, "/stub", StubModel())

    handle.set_identity(shape=np.ones(3, dtype=np.float32))

    assert np.all(handle.params["shape"] == 1.0)
    broadcast = messages[-1]
    assert isinstance(broadcast, _runtime.BodyModelsViserIdentityMessage)


def test_property_setters_route_to_updates(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())
    posed = np.ones((2, 3), dtype=np.float32)

    handle.body_pose = posed

    message = state_of(scene).models["/stub"]
    np.testing.assert_array_equal(message.skinning_transforms[:, :3, 3], posed)


def test_invalid_keys_raise(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())
    with pytest.raises(ValueError):
        handle.set_pose(bogus=np.zeros(1))
    with pytest.raises(ValueError):
        handle.set_identity(bogus=np.zeros(1))
    with pytest.raises(ValueError):
        handle.set_transform(bogus=np.zeros(1))


def test_set_transform_records_transform_only_message(scene, monkeypatch):
    messages = []
    monkeypatch.setattr(_runtime, "broadcast", lambda scene, message: messages.append(message))
    handle = bmv.add_body_model(scene, "/stub", StubModel())

    handle.set_transform(global_translation=np.array([1.0, 2.0, 3.0]))

    message = state_of(scene).models["/stub"]
    np.testing.assert_array_equal(message.global_translation, [1.0, 2.0, 3.0])
    broadcast = messages[-1]
    assert isinstance(broadcast, _runtime.BodyModelsViserTransformMessage)


def test_replay_pushes_assets_before_models(scene):
    bmv.add_body_model(scene, "/stub", StubModel())

    client_state = FakeClientState()
    _runtime._replay_state(client_state, state_of(scene))

    types = [type(message).__name__ for message in client_state.message_buffer.messages]
    assert types == ["BodyModelsViserAssetMessage", "BodyModelsViserModelMessage"]


def test_remove_clears_model_and_unused_asset(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())
    handle.remove()

    assert "/stub" not in state_of(scene).models
    assert state_of(scene).assets == {}
    client_state = FakeClientState()
    _runtime._replay_state(client_state, state_of(scene))
    assert client_state.message_buffer.messages == []
    with pytest.raises(KeyError):
        handle.remove()


def test_shared_asset_lives_until_last_model_is_removed(scene):
    model = StubModel()
    first = bmv.add_body_model(scene, "/first", model)
    second = bmv.add_body_model(scene, "/second", model)

    first.remove()
    assert len(state_of(scene).assets) == 1

    second.remove()
    assert state_of(scene).assets == {}
