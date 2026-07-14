"""Transport layer: injects the JS/WASM runtime into viser and broadcasts
body-model messages.

Everything in this module touches viser PRIVATE internals (message
serializer, websock client state, RunJavascriptMessage). It is the only
module allowed to; see "viser compatibility" in the README.
"""

from __future__ import annotations

import base64
import dataclasses
import weakref
from collections.abc import Callable
from importlib.resources import files
from typing import Any

import numpy as np
import numpy.typing as npt
from viser import _messages

from ._client_autobuild import ensure_client_is_built


@dataclasses.dataclass
class BodyModelsViserAssetMessage(_messages.Message, include_in_scene_serialization=True):
    asset_id: int
    faces: npt.NDArray[np.uint32]
    skin_weight_offsets: npt.NDArray[np.uint32]
    skin_weight_indices: npt.NDArray[np.uint16]
    skin_weight_values: npt.NDArray[np.float32]
    corrective_basis: npt.NDArray[np.int16] | None
    corrective_scales: npt.NDArray[np.float32] | None


@dataclasses.dataclass
class BodyModelsViserModelMessage(_messages.Message, include_in_scene_serialization=True):
    name: str
    asset_id: int
    rest_vertices: npt.NDArray[np.float32]
    skinning_transforms: npt.NDArray[np.float32]
    pose_coefficients: npt.NDArray[np.float32] | None
    global_rotation: npt.NDArray[np.float32]
    global_translation: npt.NDArray[np.float32]
    props: dict[str, Any]


@dataclasses.dataclass
class BodyModelsViserIdentityMessage(_messages.Message, include_in_scene_serialization=True):
    name: str
    rest_vertices: npt.NDArray[np.float32]
    skinning_transforms: npt.NDArray[np.float32]
    pose_coefficients: npt.NDArray[np.float32] | None


@dataclasses.dataclass
class BodyModelsViserPoseMessage(_messages.Message, include_in_scene_serialization=True):
    name: str
    skinning_transforms: npt.NDArray[np.float32]
    pose_coefficients: npt.NDArray[np.float32] | None


@dataclasses.dataclass
class BodyModelsViserTransformMessage(_messages.Message, include_in_scene_serialization=True):
    name: str
    global_rotation: npt.NDArray[np.float32]
    global_translation: npt.NDArray[np.float32]


@dataclasses.dataclass
class BodyModelsViserReadyMessage(_messages.Message, include_in_scene_serialization=False):
    pass


@dataclasses.dataclass
class _AssetRecord:
    message: BodyModelsViserAssetMessage
    refcount: int = 1


@dataclasses.dataclass
class RuntimeState:
    """Per-scene registry of body models, replayed to late-joining clients."""

    assets: dict[tuple[Any, bool], _AssetRecord] = dataclasses.field(default_factory=dict)
    models: dict[str, BodyModelsViserModelMessage] = dataclasses.field(default_factory=dict)
    next_asset_id: int = 1
    ready_clients: set[int] = dataclasses.field(default_factory=set)
    installed_clients: set[int] = dataclasses.field(default_factory=set)
    initialized_serializers: weakref.WeakSet[Any] = dataclasses.field(
        default_factory=weakref.WeakSet
    )


def get_state(scene: Any) -> RuntimeState:
    """Return the scene's runtime state, installing all hooks on first use."""
    websock = scene._websock_interface
    state = getattr(websock, "_body_models_viser", None)
    if state is not None:
        return state

    state = RuntimeState()
    websock._body_models_viser = state
    _install_serializer_hook(websock, state)

    def ready(client_id: int, _: BodyModelsViserReadyMessage) -> None:
        state.ready_clients.add(client_id)
        _replay_state(websock._client_state_from_id[client_id], state)

    websock.register_handler(BodyModelsViserReadyMessage, ready)
    websock.on_client_connect(lambda client: _on_client_connect(websock, state, client))
    websock.on_client_disconnect(lambda client: _on_client_disconnect(state, client))
    for client_id in websock._client_state_from_id:
        _install_client(websock, state, client_id)
    return state


def broadcast(scene: Any, message: _messages.Message) -> None:
    """Record the message into active recordings and send it to every client
    that has the runtime installed."""
    websock = scene._websock_interface
    state = get_state(scene)
    for serializer in websock._record_handles:
        _ensure_serializer_runtime(serializer, state)
        serializer._insert_message(message)
    for client_id in state.ready_clients:
        client_state = websock._client_state_from_id.get(client_id)
        if client_state is not None:
            client_state.message_buffer.push(message)


def _on_client_connect(websock: Any, state: RuntimeState, client: Any) -> None:
    _install_client(websock, state, client.client_id)


def _on_client_disconnect(state: RuntimeState, client: Any) -> None:
    state.ready_clients.discard(client.client_id)
    state.installed_clients.discard(client.client_id)


def _install_client(websock: Any, state: RuntimeState, client_id: int) -> None:
    if client_id in state.installed_clients:
        return
    state.installed_clients.add(client_id)
    client_state = websock._client_state_from_id[client_id]
    client_state.message_buffer.push(_messages.RunJavascriptMessage(_install_javascript()))
    client_state.message_buffer.flush()


def _replay_state(client_state: Any, state: RuntimeState) -> None:
    for asset in state.assets.values():
        client_state.message_buffer.push(asset.message)
    for message in state.models.values():
        client_state.message_buffer.push(message)


def _install_serializer_hook(websock: Any, state: RuntimeState) -> None:
    original_get_message_serializer = websock.get_message_serializer

    def get_message_serializer(filter: Callable[[_messages.Message], bool]) -> Any:
        serializer = original_get_message_serializer(filter)
        if state.models:
            _ensure_serializer_runtime(serializer, state)
        for asset in state.assets.values():
            serializer._insert_message(asset.message)
        for message in state.models.values():
            serializer._insert_message(message)
        return serializer

    websock.get_message_serializer = get_message_serializer


def _ensure_serializer_runtime(serializer: Any, state: RuntimeState) -> None:
    if serializer in state.initialized_serializers:
        return
    state.initialized_serializers.add(serializer)
    _install_serializer_preload(serializer)
    serializer._insert_message(_messages.RunJavascriptMessage(_install_javascript()))


def _install_serializer_preload(serializer: Any) -> None:
    original_as_html = serializer.as_html

    def as_html(*args: Any, **kwargs: Any) -> str:
        html = original_as_html(*args, **kwargs)
        head_start = html.index("<head>") + len("<head>")
        return html[:head_start] + _preload_javascript() + html[head_start:]

    serializer.as_html = as_html


def _install_javascript() -> str:
    ensure_client_is_built()
    source = (files(__package__) / "client" / "body-models-viser.js").read_text()
    wasm_path = files(__package__) / "client" / "body-models-viser.wasm"
    wasm = base64.b64encode(wasm_path.read_bytes()).decode("ascii")
    return f"""
(() => {{
  if (window.BodyModelsViser !== undefined) {{
    window.BodyModelsViser.ready();
    return;
  }}
  {source}
  window.BodyModelsViser = BodyModelsViser;
  window.BodyModelsViser.install({wasm!r});
}})();
"""


# Offline HTML exports replay the recorded messages into the viewer's queue
# before the RunJavascriptMessage carrying our runtime has executed, so
# body-model messages would be dropped as unknown types. This <head> snippet
# intercepts them by patching Array.prototype.push — a PAGE-GLOBAL patch that
# stays active until the runtime's drainPreload() (client/src/index.ts) calls
# restore(). Nothing else may restore or extend it. This cannot be scoped
# further without viser exposing an extension point for custom messages.
def _preload_javascript() -> str:
    return """
<script>
(() => {
  const key = "__BODY_MODELS_VISER_PRELOAD__";
  const pending = [];
  const originalPush = Array.prototype.push;
  const bodyModelMessageTypes = new Set([
    "BodyModelsViserAssetMessage",
    "BodyModelsViserModelMessage",
    "BodyModelsViserIdentityMessage",
    "BodyModelsViserPoseMessage",
    "BodyModelsViserTransformMessage",
  ]);
  const isBodyModelMessage = (message) =>
    message !== null &&
    typeof message === "object" &&
    bodyModelMessageTypes.has(message.type);
  const push = function (...messages) {
    const bodyModelMessages = messages.filter(isBodyModelMessage);
    const viserMessages = messages.filter((message) => !isBodyModelMessage(message));
    originalPush.apply(pending, bodyModelMessages);
    return originalPush.apply(this, viserMessages);
  };
  Array.prototype.push = push;
  window[key] = {
    pending,
    restore() {
      if (Array.prototype.push === push) {
        Array.prototype.push = originalPush;
      }
    },
  };
})();
</script>
"""
