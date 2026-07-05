from __future__ import annotations

import numpy as np
import pytest
from conftest import StubModel

import body_models_viser as bmv
from body_models_viser._client_autobuild import ASSETS

needs_client = pytest.mark.skipif(
    not all(asset.exists() for asset in ASSETS),
    reason="client bundle not built; run `cd client && npm test`",
)


@needs_client
def test_serializer_replays_runtime_then_state(scene):
    handle = bmv.add_body_model(scene, "/stub", StubModel())
    handle.set_pose(body_pose=np.ones((2, 3), dtype=np.float32))

    serializer = scene._websock_interface.get_message_serializer(lambda message: True)

    types = [payload["type"] for _, payload in serializer._messages]
    assert types[0] == "RunJavascriptMessage"
    assert "BodyModelsViserModelMessage" in types
    assert "BodyModelsViserPoseMessage" in types

    html = serializer.as_html()
    head = html.split("</head>")[0]
    assert "__BODY_MODELS_VISER_PRELOAD__" in head
