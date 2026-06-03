from __future__ import annotations

from ._skeleton import ViserSkeletonHandle, add_skeleton
from ._viser import (
    AnnyBodyHandle,
    BodyModelHandle,
    FlameBodyHandle,
    ManoBodyHandle,
    MhrBodyHandle,
    SkelBodyHandle,
    SmplBodyHandle,
    SmplhBodyHandle,
    SmplxBodyHandle,
    SomaBodyHandle,
    add_body_model,
)

__version__ = "0.1.0"

__all__ = [
    "AnnyBodyHandle",
    "BodyModelHandle",
    "FlameBodyHandle",
    "ManoBodyHandle",
    "MhrBodyHandle",
    "SkelBodyHandle",
    "SmplBodyHandle",
    "SmplhBodyHandle",
    "SmplxBodyHandle",
    "SomaBodyHandle",
    "ViserSkeletonHandle",
    "add_body_model",
    "add_skeleton",
]
