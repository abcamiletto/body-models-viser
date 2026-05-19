from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from ._viser import MhrBodyHandle, SmplBodyHandle, ViserBodyHandle, add_body_model

__version__ = "0.1.0"

__all__ = [
    "MhrBodyHandle",
    "SmplBodyHandle",
    "ViserBodyHandle",
    "add_body_model",
    "client_path",
]


def client_path() -> Path:
    return Path(str(files(__name__) / "client" / "body-models-viser.js"))
