from __future__ import annotations

from importlib.resources import files
from pathlib import Path

__version__ = "0.1.0"


def client_path() -> Path:
    return Path(str(files(__name__) / "client" / "body-models-viser.js"))
