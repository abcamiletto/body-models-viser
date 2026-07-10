from __future__ import annotations

from importlib.metadata import version

from ._body_model import BodyModelHandle, add_body_model
from ._rigid_body import ViserRigidBodyModelHandle, add_rigid_body_model
from ._skeleton import ViserSkeletonHandle, add_skeleton

__version__ = version("body-models-viser")

__all__ = [
    "BodyModelHandle",
    "ViserRigidBodyModelHandle",
    "ViserSkeletonHandle",
    "add_body_model",
    "add_rigid_body_model",
    "add_skeleton",
]
