from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Float
from nanomanifold import SO3

if TYPE_CHECKING:
    import viser


def _pose_property(key: str) -> property:
    def get(self: ViserRigidBodyModelHandle) -> np.ndarray:
        return self.pose[key]

    def set(self: ViserRigidBodyModelHandle, value: np.ndarray) -> None:
        self.set_pose(**{key: value})

    return property(get, set)


class ViserRigidBodyModelHandle:
    """Rigid articulated body model rendered as one static mesh per link."""

    body_pose = _pose_property("body_pose")
    hand_pose = _pose_property("hand_pose")
    global_rotation = _pose_property("global_rotation")
    global_translation = _pose_property("global_translation")

    def __init__(
        self,
        model: Any,
        pose: dict[str, Float[np.ndarray, "..."]],
        root_frame: viser.FrameHandle,
        links: list[viser.MeshHandle],
    ) -> None:
        self.model = model
        self.model_name = model.__class__.__name__
        self.pose = pose
        self.root_frame = root_frame
        self.links = links

    @property
    def name(self) -> str:
        return self.root_frame.name

    @property
    def wxyz(self) -> Float[np.ndarray, "4"]:
        return self.root_frame.wxyz

    @wxyz.setter
    def wxyz(self, value: tuple[float, float, float, float] | np.ndarray) -> None:
        self.root_frame.wxyz = np.asarray(value)

    @property
    def position(self) -> Float[np.ndarray, "3"]:
        return self.root_frame.position

    @position.setter
    def position(self, value: tuple[float, float, float] | np.ndarray) -> None:
        self.root_frame.position = np.asarray(value)

    @property
    def visible(self) -> bool:
        return self.root_frame.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self.root_frame.visible = value

    def set_pose(self, **forward_kwargs: Float[np.ndarray, "..."] | np.ndarray) -> None:
        invalid = forward_kwargs.keys() - self.pose.keys()
        if invalid:
            raise ValueError(f"{self.model_name} does not support: {', '.join(sorted(invalid))}.")
        for name, value in forward_kwargs.items():
            self.pose[name] = np.asarray(value).copy()
        self._apply_pose()

    def _apply_pose(self) -> None:
        transforms = np.asarray(self.model.forward_links(**self.pose))
        rotations = transforms[:, :3, :3]
        wxyzs = SO3.conversions.from_rotmat_to_quat(rotations, convention="wxyz", xp=np)
        positions = transforms[:, :3, 3]
        for handle, wxyz, position in zip(self.links, wxyzs, positions):
            handle.wxyz = wxyz
            handle.position = position

    def remove(self) -> None:
        for handle in self.links:
            handle.remove()
        self.root_frame.remove()


def add_rigid_body_model(
    scene: viser.SceneApi,
    name: str,
    model: Any,
    *,
    color: tuple[float, float, float] = (180, 180, 180),
) -> ViserRigidBodyModelHandle:
    """Add a rigid articulated body model to a ``viser`` scene."""
    if not model.is_rigid_body:
        model_name = model.__class__.__name__
        raise ValueError(f"add_rigid_body_model() only supports rigid models, got {model_name}.")

    pose = model.get_rest_pose()
    root = scene.add_frame(name, show_axes=False)

    links = []
    for index, link_name in enumerate(model.link_names):
        mesh = model.link_mesh(link_name)
        link_path = f"{name}/links/{index:03d}"
        links.append(
            scene.add_mesh_simple(
                link_path,
                vertices=np.asarray(mesh["vertices"], dtype=np.float32),
                faces=np.asarray(mesh["faces"]),
                color=color,
            )
        )
    handle = ViserRigidBodyModelHandle(model, pose, root, links)
    handle._apply_pose()
    return handle
