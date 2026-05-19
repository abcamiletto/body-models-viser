from __future__ import annotations

from typing import Any

import numpy as np


class ViserBodyHandle:
    """Non-rigid body model rendered as a full vertex mesh."""

    def __init__(self, model: Any, pose: dict[str, np.ndarray], root_frame: Any, mesh: Any) -> None:
        self.model = model
        self.model_name = model.__class__.__name__
        self.pose = pose
        self.root_frame = root_frame
        self.mesh = mesh

    @property
    def name(self) -> str:
        return self.root_frame.name

    @property
    def wxyz(self) -> np.ndarray:
        return self.root_frame.wxyz

    @wxyz.setter
    def wxyz(self, value: tuple[float, float, float, float] | np.ndarray) -> None:
        value = np.asarray(value)
        assert value.shape == (4,)
        self.root_frame.wxyz = value

    @property
    def position(self) -> np.ndarray:
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
        self.mesh.visible = value

    @property
    def shape(self) -> np.ndarray:
        return self._param("shape")

    @shape.setter
    def shape(self, value: np.ndarray) -> None:
        self.set_pose(shape=value)

    @property
    def body_pose(self) -> np.ndarray:
        return self._param("body_pose")

    @body_pose.setter
    def body_pose(self, value: np.ndarray) -> None:
        self.set_pose(body_pose=value)

    @property
    def hand_pose(self) -> np.ndarray:
        return self._param("hand_pose")

    @hand_pose.setter
    def hand_pose(self, value: np.ndarray) -> None:
        self.set_pose(hand_pose=value)

    @property
    def head_pose(self) -> np.ndarray:
        return self._param("head_pose")

    @head_pose.setter
    def head_pose(self, value: np.ndarray) -> None:
        self.set_pose(head_pose=value)

    @property
    def expression(self) -> np.ndarray:
        return self._param("expression")

    @expression.setter
    def expression(self, value: np.ndarray) -> None:
        self.set_pose(expression=value)

    @property
    def global_rotation(self) -> np.ndarray:
        return self._param("global_rotation")

    @global_rotation.setter
    def global_rotation(self, value: np.ndarray) -> None:
        self.set_pose(global_rotation=value)

    @property
    def global_translation(self) -> np.ndarray:
        return self._param("global_translation")

    @global_translation.setter
    def global_translation(self, value: np.ndarray) -> None:
        self.set_pose(global_translation=value)

    @property
    def pelvis_rotation(self) -> np.ndarray:
        return self._param("pelvis_rotation")

    @pelvis_rotation.setter
    def pelvis_rotation(self, value: np.ndarray) -> None:
        self.set_pose(pelvis_rotation=value)

    def set_pose(self, **forward_kwargs: np.ndarray) -> None:
        changed = False
        for name, value in forward_kwargs.items():
            current = self._param(name)
            value = np.asarray(value)
            if np.array_equal(current, value):
                continue
            self.pose[name] = value.copy()
            changed = True
        if changed:
            self._apply_pose()

    def remove(self) -> None:
        self.mesh.remove()
        self.root_frame.remove()

    def _param(self, name: str) -> np.ndarray:
        assert name in self.pose, f"{self.model_name} does not support {name!r}."
        return self.pose[name]

    def _apply_pose(self) -> None:
        self.mesh.vertices = _vertices(self.model, self.pose)


class SmplBodyHandle(ViserBodyHandle):
    pass


class MhrBodyHandle(ViserBodyHandle):
    pass


def add_body_model(
    scene: Any,
    name: str,
    model: Any,
    *,
    color: tuple[float, float, float] = (180, 180, 180),
) -> ViserBodyHandle:
    """Add a non-rigid SMPL or MHR body model to a viser scene."""
    if getattr(model, "is_rigid_body", False):
        raise ValueError("add_body_model() only supports non-rigid models.")

    pose = {key: np.asarray(value).copy() for key, value in model.get_rest_pose().items()}
    root = scene.add_frame(name, show_axes=False)
    mesh = scene.add_mesh_simple(
        f"{name}/mesh",
        vertices=_vertices(model, pose),
        faces=_triangular_faces(model),
        color=color,
    )
    handle_type = _handle_type(model)
    return handle_type(model, pose, root, mesh)


def _handle_type(model: Any) -> type[ViserBodyHandle]:
    model_name = model.__class__.__name__.lower()
    if model_name == "smpl":
        return SmplBodyHandle
    if model_name == "mhr":
        return MhrBodyHandle
    raise ValueError(f"Unsupported body model {model.__class__.__name__!r}.")


def _vertices(model: Any, pose: dict[str, np.ndarray]) -> np.ndarray:
    return _as_unbatched_array(model.forward_vertices(**pose))


def _as_unbatched_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return value[0] if value.ndim >= 3 and value.shape[0] == 1 else value


def _triangular_faces(model: Any) -> np.ndarray:
    faces = np.asarray(model.faces, dtype=np.uint32)
    if faces.shape[1] == 3:
        return faces
    if faces.shape[1] == 4:
        return np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
    raise ValueError(f"Expected triangular or quad faces, got {faces.shape}.")
