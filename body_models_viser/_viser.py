from __future__ import annotations

import dataclasses
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from viser import _messages


@dataclasses.dataclass
class BodyModelMeshMessage(_messages.Message, include_in_scene_serialization=True):
    name: str
    vertices: list[list[float]]
    faces: list[list[int]]
    skinWeights: list[list[float]]
    skinJoints: list[list[int]]
    boneTransforms: list[list[list[float]]]
    color: tuple[int, int, int]
    wireframe: bool
    opacity: float | None
    flatShading: bool
    side: str
    material: str
    scale: float | tuple[float, float, float]
    castShadow: bool
    receiveShadow: bool | float


@dataclasses.dataclass
class BodyModelPoseMessage(_messages.Message, include_in_scene_serialization=True):
    name: str
    vertices: list[list[float]]
    boneTransforms: list[list[list[float]]]


class ViserBodyHandle:
    """Non-rigid body model driven by the body-models-viser browser runtime."""

    def __init__(
        self,
        scene: Any,
        name: str,
        model: Any,
        pose: dict[str, np.ndarray],
    ) -> None:
        self.scene = scene
        self._name = name
        self.model = model
        self.model_name = model.__class__.__name__
        self.pose = pose
        self._wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self._position = np.zeros(3)
        self._visible = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def wxyz(self) -> np.ndarray:
        return self._wxyz

    @wxyz.setter
    def wxyz(self, value: tuple[float, float, float, float] | np.ndarray) -> None:
        value = np.asarray(value)
        assert value.shape == (4,)
        self._wxyz = value.astype(float, copy=True)
        message = _messages.SetOrientationMessage(self.name, tuple(self._wxyz))
        self.scene._websock_interface.queue_message(message)

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: tuple[float, float, float] | np.ndarray) -> None:
        value = np.asarray(value)
        assert value.shape == (3,)
        self._position = value.astype(float, copy=True)
        message = _messages.SetPositionMessage(self.name, tuple(self._position))
        self.scene._websock_interface.queue_message(message)

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = bool(value)
        message = _messages.SetSceneNodeVisibilityMessage(self.name, self._visible)
        self.scene._websock_interface.queue_message(message)

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
        self.scene._websock_interface.queue_message(_messages.RemoveSceneNodeMessage(self.name))

    def _param(self, name: str) -> np.ndarray:
        assert name in self.pose, f"{self.model_name} does not support {name!r}."
        return self.pose[name]

    def _apply_pose(self) -> None:
        vertices, bone_transforms = _skinning_state(self.model, self.pose)
        self.scene._websock_interface.queue_message(
            BodyModelPoseMessage(
                self.name,
                np.asarray(vertices, dtype=np.float32).tolist(),
                np.asarray(bone_transforms, dtype=np.float32).tolist(),
            ),
        )


class SmplBodyHandle(ViserBodyHandle):
    pass


class MhrBodyHandle(ViserBodyHandle):
    pass


def add_body_model(
    scene: Any,
    name: str,
    model: Any,
    *,
    color: tuple[int, int, int] = (180, 180, 180),
    wireframe: bool = False,
    opacity: float | None = None,
    flat_shading: bool = False,
    side: str = "front",
    material: str = "standard",
    scale: float | tuple[float, float, float] = 1.0,
    cast_shadow: bool = True,
    receive_shadow: bool | float = True,
) -> ViserBodyHandle:
    """Add a non-rigid SMPL or MHR body model to a viser scene."""
    if getattr(model, "is_rigid_body", False):
        raise ValueError("add_body_model() only supports non-rigid models.")

    pose = {key: np.asarray(value).copy() for key, value in model.get_rest_pose().items()}
    _install_runtime(scene)
    vertices, bone_transforms = _skinning_state(model, pose)
    skin_weights, skin_joints = _sparse_skinning(model)
    scene._websock_interface.queue_message(
        BodyModelMeshMessage(
            name=name,
            vertices=np.asarray(vertices, dtype=np.float32).tolist(),
            faces=np.asarray(_triangular_faces(model), dtype=np.int32).tolist(),
            skinWeights=np.asarray(skin_weights, dtype=np.float32).tolist(),
            skinJoints=np.asarray(skin_joints, dtype=np.int32).tolist(),
            boneTransforms=np.asarray(bone_transforms, dtype=np.float32).tolist(),
            color=color,
            wireframe=wireframe,
            opacity=opacity,
            flatShading=flat_shading,
            side=side,
            material=material,
            scale=scale,
            castShadow=cast_shadow,
            receiveShadow=receive_shadow,
        ),
    )
    handle_type = _handle_type(model)
    return handle_type(scene, name, model, pose)


def _handle_type(model: Any) -> type[ViserBodyHandle]:
    model_name = model.__class__.__name__.lower()
    if model_name == "smpl":
        return SmplBodyHandle
    if model_name == "mhr":
        return MhrBodyHandle
    raise ValueError(f"Unsupported body model {model.__class__.__name__!r}.")


def _install_runtime(scene: Any) -> None:
    scene._websock_interface.queue_message(_messages.RunJavascriptMessage(_runtime_source()))


def _runtime_source() -> str:
    packaged = files(__package__) / "client" / "body-models-viser.js"
    if packaged.is_file():
        return packaged.read_text()

    development = Path(__file__).resolve().parents[1] / "client" / "dist" / "body-models-viser.js"
    return development.read_text()


def _skinning_state(model: Any, pose: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    vertices = _unskinned_vertices(model, pose)
    bone_transforms = _as_unbatched_array(model.forward_skeleton(**pose))
    return vertices, bone_transforms


def _unskinned_vertices(model: Any, pose: dict[str, np.ndarray]) -> np.ndarray:
    vertices = np.asarray(model.rest_vertices, dtype=np.float32)
    shape = pose.get("shape")
    if shape is not None and hasattr(model, "shapedirs"):
        shape = _as_unbatched_array(shape)
        vertices = vertices + np.einsum("s,vcs->vc", shape, model.shapedirs[..., : shape.shape[-1]])
    return vertices


def _as_unbatched_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return value[0] if value.ndim >= 2 and value.shape[0] == 1 else value


def _triangular_faces(model: Any) -> np.ndarray:
    faces = np.asarray(model.faces, dtype=np.uint32)
    if faces.shape[1] == 3:
        return faces
    if faces.shape[1] == 4:
        return np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
    raise ValueError(f"Expected triangular or quad faces, got {faces.shape}.")


def _sparse_skinning(model: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model.weights, "skin_indices") and hasattr(model.weights, "skin_weights"):
        return model.weights.skin_weights, model.weights.skin_indices

    dense = np.asarray(model.skin_weights)
    skin_joints: list[np.ndarray] = []
    skin_weights: list[np.ndarray] = []
    for row in dense:
        joints = np.flatnonzero(row)
        skin_joints.append(joints)
        skin_weights.append(row[joints])
    max_len = max(len(row) for row in skin_joints)
    joints_out = np.zeros((len(skin_joints), max_len), dtype=np.int32)
    weights_out = np.zeros((len(skin_weights), max_len), dtype=np.float32)
    for vertex, (joints, weights) in enumerate(zip(skin_joints, skin_weights)):
        joints_out[vertex, : len(joints)] = joints
        weights_out[vertex, : len(weights)] = weights
    return weights_out, joints_out
