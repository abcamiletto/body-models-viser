from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

__version__ = "0.1.0"


def client_path() -> Path:
    return Path(str(files(__name__) / "client" / "body-models-viser.js"))


def add_body_model(*args: Any, **kwargs: Any) -> Any:
    from body_models.extras.viser_plugin import add_body_model as add

    return add(*args, **kwargs)


def patch_viser(viser_module: Any | None = None) -> None:
    if viser_module is None:
        try:
            import viser as viser_module
        except ModuleNotFoundError:
            return

    scene_api = getattr(viser_module, "SceneApi", None)
    if scene_api is None or hasattr(scene_api, "add_body_model"):
        return
    scene_api.add_body_model = add_body_model
    scene_api.addBodyModel = add_body_model


patch_viser()
