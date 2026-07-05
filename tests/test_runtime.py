from __future__ import annotations

import pytest
from conftest import FakeClientState, StubModel

import body_models_viser as bmv
from body_models_viser import _runtime
from body_models_viser._client_autobuild import ASSETS

needs_client = pytest.mark.skipif(
    not all(asset.exists() for asset in ASSETS),
    reason="client bundle not built; run `cd client && npm test`",
)


class FakeClient:
    def __init__(self, client_id):
        self.client_id = client_id


@needs_client
def test_client_connect_installs_runtime_once(scene):
    bmv.add_body_model(scene, "/stub", StubModel())
    websock = scene._websock_interface
    state = _runtime.get_state(scene)
    client_state = FakeClientState()
    websock._client_state_from_id[999] = client_state
    try:
        _runtime._on_client_connect(websock, state, FakeClient(999))
        _runtime._on_client_connect(websock, state, FakeClient(999))
    finally:
        del websock._client_state_from_id[999]

    assert 999 in state.installed_clients
    types = [type(message).__name__ for message in client_state.message_buffer.messages]
    assert types == ["RunJavascriptMessage"]


def test_client_disconnect_prunes_state(scene):
    bmv.add_body_model(scene, "/stub", StubModel())
    state = _runtime.get_state(scene)
    state.installed_clients.add(7)
    state.ready_clients.add(7)

    _runtime._on_client_disconnect(state, FakeClient(7))

    assert 7 not in state.installed_clients
    assert 7 not in state.ready_clients
