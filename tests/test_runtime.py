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
def test_client_connect_installs_runtime(scene):
    bmv.add_body_model(scene, "/stub", StubModel())
    websock = scene._websock_interface
    client_state = FakeClientState()
    websock._client_state_from_id[999] = client_state
    try:
        _runtime._install_client(websock, 999)
    finally:
        del websock._client_state_from_id[999]

    types = [type(message).__name__ for message in client_state.message_buffer.messages]
    assert types == ["RunJavascriptMessage"]


def test_client_disconnect_prunes_state(scene):
    bmv.add_body_model(scene, "/stub", StubModel())
    state = _runtime.get_state(scene)
    state.ready_clients.add(7)

    scene._websock_interface._client_disconnect_cb[-1](FakeClient(7))

    assert 7 not in state.ready_clients
