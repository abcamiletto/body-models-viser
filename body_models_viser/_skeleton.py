from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
from jaxtyping import Float, Int

if TYPE_CHECKING:
    import viser


class ViserSkeletonHandle:
    """Skeleton rendered as joint markers and cylinder bones."""

    def __init__(
        self,
        root_frame: viser.FrameHandle,
        parent_tree: list[tuple[int, int]],
        bones: viser.BatchedMeshHandle,
        bone_radius: float,
        joint_positions: Float[np.ndarray, "N 3"],
        joints: viser.BatchedMeshHandle,
    ) -> None:
        self.root_frame = root_frame
        self.parent_tree = parent_tree
        self.bones = bones
        self.bone_radius = bone_radius
        self._joint_positions = joint_positions
        self.joints = joints

    @property
    def name(self) -> str:
        return self.root_frame.name

    @property
    def wxyz(self) -> Float[np.ndarray, "4"]:
        return self.root_frame.wxyz

    @wxyz.setter
    def wxyz(self, value: tuple[float, float, float, float] | np.ndarray) -> None:
        value = np.asarray(value)
        assert value.shape == (4,)
        self.root_frame.wxyz = value

    @property
    def position(self) -> Float[np.ndarray, "3"]:
        return self.root_frame.position

    @position.setter
    def position(self, value: tuple[float, float, float] | np.ndarray) -> None:
        value = np.asarray(value)
        assert value.shape == (3,)
        self.root_frame.position = value

    @property
    def visible(self) -> bool:
        return self.root_frame.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self.root_frame.visible = value

    @property
    def joint_positions(self) -> Float[np.ndarray, "N 3"]:
        return self._joint_positions

    @joint_positions.setter
    def joint_positions(self, value: Float[np.ndarray, "N 3"]) -> None:
        self._joint_positions = np.asarray(value, dtype=np.float32)
        positions, wxyzs, scales = _bone_instances(
            self._joint_positions, self.parent_tree, self.bone_radius
        )
        self.joints.batched_positions = self._joint_positions
        self.bones.batched_positions = positions
        self.bones.batched_wxyzs = wxyzs
        self.bones.batched_scales = scales

    def remove(self) -> None:
        self.root_frame.remove()


def add_skeleton(
    scene: viser.SceneApi,
    name: str,
    joint_positions: Float[np.ndarray, "N 3"],
    parents: Int[np.ndarray, "N"] | list[int] | tuple[int, ...],
    *,
    joint_names: tuple[str, ...] | None = None,
    color: tuple[float, float, float] = (120, 180, 255),
    joint_color: tuple[float, float, float] = (255, 255, 255),
    bone_radius: float = 0.006,
    joint_radius: float = 0.015,
) -> ViserSkeletonHandle:
    """Add a clickable skeleton to a ``viser`` scene."""
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    parents = [int(parent) for parent in parents]
    joint_names = joint_names or tuple(
        f"joint_{index}" for index in range(len(joint_positions))
    )
    parent_tree = [
        (parent, index) for index, parent in enumerate(parents) if parent >= 0
    ]
    root = scene.add_frame(name, show_axes=False)

    bone_positions, bone_wxyzs, bone_scales = _bone_instances(
        joint_positions, parent_tree, bone_radius
    )
    cylinder = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=16)
    bones = scene.add_batched_meshes_simple(
        f"{name}/bones",
        cylinder.vertices,
        cylinder.faces,
        bone_wxyzs,
        bone_positions,
        batched_scales=bone_scales,
        batched_colors=color,
    )

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    joint_wxyzs = np.zeros((len(joint_positions), 4), dtype=np.float32)
    joint_wxyzs[:, 0] = 1.0
    joints = scene.add_batched_meshes_simple(
        f"{name}/joints",
        sphere.vertices,
        sphere.faces,
        joint_wxyzs,
        joint_positions,
        batched_scales=np.full(len(joint_positions), joint_radius, dtype=np.float32),
        batched_colors=joint_color,
    )

    @joints.on_click
    def _(event) -> None:
        assert event.instance_index is not None
        event.client.add_notification(
            title=f"Joint: {joint_names[event.instance_index]}",
            body="",
            auto_close_seconds=3,
        )

    return ViserSkeletonHandle(
        root, parent_tree, bones, bone_radius, joint_positions, joints
    )


def _bone_instances(
    joint_positions: Float[np.ndarray, "N 3"],
    parent_tree: list[tuple[int, int]],
    radius: float,
) -> tuple[
    Float[np.ndarray, "B 3"],
    Float[np.ndarray, "B 4"],
    Float[np.ndarray, "B 3"],
]:
    positions = np.empty((len(parent_tree), 3), dtype=np.float32)
    wxyzs = np.empty((len(parent_tree), 4), dtype=np.float32)
    scales = np.full((len(parent_tree), 3), radius, dtype=np.float32)
    for index, (parent, child) in enumerate(parent_tree):
        positions[index], wxyzs[index], scales[index, 2] = _cylinder_between(
            joint_positions[parent], joint_positions[child]
        )
    return positions, wxyzs, scales


def _cylinder_between(
    p0: Float[np.ndarray, "3"],
    p1: Float[np.ndarray, "3"],
) -> tuple[Float[np.ndarray, "3"], Float[np.ndarray, "4"], float]:
    diff = p1 - p0
    height = float(np.linalg.norm(diff))
    if height == 0.0:
        wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return (p0 + p1) / 2.0, wxyz, height
    direction = diff / height
    dot = float(direction[2])

    if dot > 0.9999:
        wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    elif dot < -0.9999:
        wxyz = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = np.array([-direction[1], direction[0], 0.0], dtype=np.float32)
        axis /= np.linalg.norm(axis)
        half = np.arccos(dot) / 2.0
        wxyz = np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float32)

    return (p0 + p1) / 2.0, wxyz, height
