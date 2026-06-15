from __future__ import annotations

from ._rigid_body import ViserRigidBodyModelHandle, add_rigid_body_model
from ._skeleton import ViserSkeletonHandle, add_skeleton
from ._body_model import (
    AnnyBodyHandle,
    BodyModelHandle,
    FlameBodyHandle,
    GarmentMeasurementsBodyHandle,
    ManoBodyHandle,
    MhrBodyHandle,
    SkelBodyHandle,
    SmplBodyHandle,
    SmplhBodyHandle,
    SmplxBodyHandle,
    SomaBodyHandle,
    add_body_model,
)

__version__ = "0.3.1"

__all__ = [
    "AnnyBodyHandle",
    "BodyModelHandle",
    "FlameBodyHandle",
    "GarmentMeasurementsBodyHandle",
    "ManoBodyHandle",
    "MhrBodyHandle",
    "SkelBodyHandle",
    "SmplBodyHandle",
    "SmplhBodyHandle",
    "SmplxBodyHandle",
    "SomaBodyHandle",
    "ViserRigidBodyModelHandle",
    "ViserSkeletonHandle",
    "add_body_model",
    "add_rigid_body_model",
    "add_skeleton",
]
