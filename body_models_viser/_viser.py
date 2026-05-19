from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from body_models.anny import pose as anny_pose
from body_models.anny.backends import core as anny_core
from body_models.mhr.backends import core as mhr_core
from body_models.mhr.pose import pack_pose
from body_models.smpl.backends import core as smpl_core
from body_models.smplh.backends import core as smplh_core
from body_models.smplx.backends import core as smplx_core
from body_models.soma import pose as soma_pose
from body_models.soma.backends import core as soma_core
from jaxtyping import Float, Int
from nanomanifold import SO3
from viser import _messages

_HANDLE_TYPES: dict[str, type["ViserBodyHandle"]] = {}
_IDENTITY_PARAMS = {"gender", "age", "muscle", "weight", "height", "proportions"}
_BASE_VERTEX_PARAMS = {
    "anny": _IDENTITY_PARAMS,
    "smpl": {"shape"},
    "smplh": {"shape"},
    "smplx": {"shape", "expression"},
    "mhr": {"shape", "expression"},
    "soma": {"identity", "scale_params"},
}
_VERTEX_PARAMS = {
    "anny": _IDENTITY_PARAMS,
    "smpl": {"shape", "body_pose"},
    "smplh": {"shape", "body_pose", "hand_pose"},
    "smplx": {"shape", "body_pose", "hand_pose", "head_pose", "expression"},
    "mhr": {"shape", "body_pose", "hand_pose", "expression"},
    "soma": {"identity", "scale_params", "body_pose", "head_pose", "hand_pose", "global_rotation"},
}


class ViserBodyHandle:
    """Non-rigid body model driven by the body-models-viser browser runtime."""

    def __init__(
        self,
        scene: Any,
        name: str,
        model: Any,
        pose: dict[
            str,
            Float[np.ndarray, "dim"]
            | Float[np.ndarray, "items dim"]
            | Float[np.ndarray, "items rows cols"],
        ],
        vertices: Float[np.ndarray, "vertices 3"],
        base_vertices: Float[np.ndarray, "vertices 3"],
    ) -> None:
        self.scene = scene
        self._name = name
        self.model = model
        self.model_name = model.__class__.__name__.lower()
        self._base_vertex_params = _BASE_VERTEX_PARAMS[self.model_name]
        self._vertex_params = _VERTEX_PARAMS[self.model_name]
        self.pose = pose
        self._vertices = vertices
        self._base_vertices = base_vertices
        self._wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self._position = np.zeros(3)
        self._visible = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def wxyz(self) -> Float[np.ndarray, "4"]:
        return self._wxyz

    @wxyz.setter
    def wxyz(self, value: tuple[float, float, float, float] | Float[np.ndarray, "4"]) -> None:
        value = np.asarray(value)
        assert value.shape == (4,)
        self._wxyz = value.astype(float, copy=True)
        message = _messages.SetOrientationMessage(self.name, tuple(self._wxyz))
        self.scene._websock_interface.queue_message(message)

    @property
    def position(self) -> Float[np.ndarray, "3"]:
        return self._position

    @position.setter
    def position(self, value: tuple[float, float, float] | Float[np.ndarray, "3"]) -> None:
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
    def shape(self) -> Float[np.ndarray, "dim"]:
        return self._param("shape")

    @shape.setter
    def shape(self, value: Float[np.ndarray, "dim"]) -> None:
        self.set_pose(shape=value)

    @property
    def body_pose(self) -> Float[np.ndarray, "joints 3"]:
        return self._param("body_pose")

    @body_pose.setter
    def body_pose(self, value: Float[np.ndarray, "joints 3"]) -> None:
        self.set_pose(body_pose=value)

    @property
    def hand_pose(self) -> Float[np.ndarray, "joints 3"]:
        return self._param("hand_pose")

    @hand_pose.setter
    def hand_pose(self, value: Float[np.ndarray, "joints 3"]) -> None:
        self.set_pose(hand_pose=value)

    @property
    def head_pose(self) -> Float[np.ndarray, "joints 3"]:
        return self._param("head_pose")

    @head_pose.setter
    def head_pose(self, value: Float[np.ndarray, "joints 3"]) -> None:
        self.set_pose(head_pose=value)

    @property
    def expression(self) -> Float[np.ndarray, "dim"]:
        return self._param("expression")

    @expression.setter
    def expression(self, value: Float[np.ndarray, "dim"]) -> None:
        self.set_pose(expression=value)

    @property
    def global_rotation(self) -> Float[np.ndarray, "3"]:
        return self._param("global_rotation")

    @global_rotation.setter
    def global_rotation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_pose(global_rotation=value)

    @property
    def global_translation(self) -> Float[np.ndarray, "3"]:
        return self._param("global_translation")

    @global_translation.setter
    def global_translation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_pose(global_translation=value)

    @property
    def pelvis_rotation(self) -> Float[np.ndarray, "3"]:
        return self._param("pelvis_rotation")

    @pelvis_rotation.setter
    def pelvis_rotation(self, value: Float[np.ndarray, "3"]) -> None:
        self.set_pose(pelvis_rotation=value)

    def set_pose(
        self,
        **forward_kwargs: Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ) -> None:
        changed: set[str] = set()
        for name, value in forward_kwargs.items():
            current = self._param(name)
            value = np.asarray(value)
            if np.array_equal(current, value):
                continue
            self.pose[name] = value.copy()
            changed.add(name)
        if changed:
            self._apply_pose(changed)

    def remove(self) -> None:
        self.scene._websock_interface.queue_message(_messages.RemoveSceneNodeMessage(self.name))

    def _param(
        self, name: str
    ) -> Float[np.ndarray, "dim"] | Float[np.ndarray, "items dim"] | Float[np.ndarray, "items rows cols"]:
        assert name in self.pose, f"{self.model.__class__.__name__} does not support {name!r}."
        return self.pose[name]

    def _apply_pose(self, changed: set[str]) -> None:
        if changed & self._base_vertex_params:
            self._base_vertices = _base_vertices(self.model, self.pose)

        vertex_payload = None
        if changed & self._vertex_params:
            self._vertices = _vertices(self.model, self.pose, self._base_vertices)
            vertex_payload = np.asarray(self._vertices, dtype=np.float32).tolist()

        bone_transforms = _bone_transforms(self.model, self.pose)
        _runtime_call(
            self.scene,
            "receivePose",
            {
                "name": self.name,
                "vertices": vertex_payload,
                "boneTransforms": np.asarray(bone_transforms, dtype=np.float32).tolist(),
            },
        )


class SmplBodyHandle(ViserBodyHandle):
    pass


class SmplhBodyHandle(ViserBodyHandle):
    pass


class SmplxBodyHandle(ViserBodyHandle):
    pass


class MhrBodyHandle(ViserBodyHandle):
    pass


class AnnyBodyHandle(ViserBodyHandle):
    pass


class SomaBodyHandle(ViserBodyHandle):
    pass


_HANDLE_TYPES.update({
    "anny": AnnyBodyHandle,
    "smpl": SmplBodyHandle,
    "smplh": SmplhBodyHandle,
    "smplx": SmplxBodyHandle,
    "mhr": MhrBodyHandle,
    "soma": SomaBodyHandle,
})


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
    """Add a non-rigid body model to a viser scene."""
    if getattr(model, "is_rigid_body", False):
        raise ValueError("add_body_model() only supports non-rigid models.")

    pose = {key: np.asarray(value).copy() for key, value in model.get_rest_pose().items()}
    _install_runtime(scene)
    base_vertices = _base_vertices(model, pose)
    vertices = _vertices(model, pose, base_vertices)
    bone_transforms = _bone_transforms(model, pose)
    skin_weights, skin_joints = _sparse_skinning(model)
    _runtime_call(
        scene,
        "receiveMesh",
        {
            "name": name,
            "vertices": np.asarray(vertices, dtype=np.float32).tolist(),
            "faces": np.asarray(_triangular_faces(model), dtype=np.int32).tolist(),
            "skinWeights": np.asarray(skin_weights, dtype=np.float32).tolist(),
            "skinJoints": np.asarray(skin_joints, dtype=np.int32).tolist(),
            "boneTransforms": np.asarray(bone_transforms, dtype=np.float32).tolist(),
            "color": color,
            "wireframe": wireframe,
            "opacity": opacity,
            "flatShading": flat_shading,
            "side": side,
            "material": material,
            "scale": scale,
            "castShadow": cast_shadow,
            "receiveShadow": receive_shadow,
        },
    )
    handle_type = _handle_type(model)
    return handle_type(scene, name, model, pose, vertices, base_vertices)


def _handle_type(model: Any) -> type[ViserBodyHandle]:
    model_name = model.__class__.__name__.lower()
    try:
        return _HANDLE_TYPES[model_name]
    except KeyError:
        raise ValueError(f"Unsupported body model {model.__class__.__name__!r}.") from None


def _install_runtime(scene: Any) -> None:
    scene._websock_interface.queue_message(_messages.RunJavascriptMessage(_runtime_source()))


def _runtime_call(scene: Any, method: str, payload: dict[str, Any]) -> None:
    source = f"window.__BODY_MODELS_VISER__.{method}({json.dumps(payload, separators=(',', ':'))});"
    scene._websock_interface.queue_message(_messages.RunJavascriptMessage(source))


def _runtime_source() -> str:
    packaged = files(__package__) / "client" / "body-models-viser.js"
    if packaged.is_file():
        return packaged.read_text()

    development = Path(__file__).resolve().parents[1] / "client" / "dist" / "body-models-viser.js"
    return development.read_text()


def _base_vertices(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Float[np.ndarray, "vertices 3"]:
    model_name = model.__class__.__name__.lower()
    if model_name == "anny":
        return _anny_base_vertices(model, pose)
    if model_name == "smpl":
        return _smpl_base_vertices(model, pose)
    if model_name == "smplh":
        return _smpl_base_vertices(model, pose)
    if model_name == "smplx":
        return _smplx_base_vertices(model, pose)
    if model_name == "mhr":
        expression = pose.get("expression")
        if expression is None:
            expression = np.zeros((*pose["shape"].shape[:-1], model.EXPR_DIM), dtype=pose["shape"].dtype)
        coeffs = np.concatenate([pose["shape"], expression], axis=-1)
        vertices = model.weights.base_vertices + np.einsum(
            "...i,ivk->...vk",
            coeffs,
            model.weights.blendshape_dirs,
        )
        return _as_unbatched_array(vertices)
    if model_name == "soma":
        return _soma_prepared_identity(model, pose).bind_shape_active * 0.01
    raise ValueError(f"Unsupported body model {model.__class__.__name__!r}.")


def _vertices(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
    base_vertices: Float[np.ndarray, "vertices 3"],
) -> Float[np.ndarray, "vertices 3"]:
    model_name = model.__class__.__name__.lower()
    if model_name == "anny":
        return base_vertices
    if model_name == "smpl":
        return _smpl_vertices(model, pose, base_vertices)
    if model_name == "smplh":
        return _smplh_vertices(model, pose, base_vertices)
    if model_name == "smplx":
        return _smplx_vertices(model, pose, base_vertices)
    if model_name == "mhr":
        return _mhr_vertices(model, pose, base_vertices)
    if model_name == "soma":
        return _soma_vertices(model, pose, base_vertices)
    raise ValueError(f"Unsupported body model {model.__class__.__name__!r}.")


def _bone_transforms(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Float[np.ndarray, "joints 4 4"]:
    model_name = model.__class__.__name__.lower()
    if model_name == "anny":
        return _anny_bone_transforms(model, pose)
    if model_name == "smpl":
        _, joints, _, _ = smpl_core._forward_core(
            xp=np,
            v_template=None,
            shapedirs=None,
            j_template=model.weights.j_template,
            j_shapedirs=model.weights.j_shapedirs,
            parents=model.weights.parents,
            kinematic_fronts=model.weights.kinematic_fronts,
            shape=pose["shape"],
            body_pose=pose["body_pose"],
            pelvis_rotation=pose.get("pelvis_rotation"),
            skeleton_only=True,
            rotation_type=model.rotation_type,
        )
        return _smpl_lbs_transforms(model.forward_skeleton(**pose), joints)
    if model_name == "smplh":
        _, joints, _, _ = smplh_core._forward_core(
            xp=np,
            v_template=None,
            shapedirs=None,
            j_template=model.weights.j_template,
            j_shapedirs=model.weights.j_shapedirs,
            parents=model.weights.parents,
            kinematic_fronts=model.weights.kinematic_fronts,
            hand_mean=model.weights.hand_mean,
            shape=pose["shape"],
            body_pose=pose["body_pose"],
            hand_pose=pose["hand_pose"],
            pelvis_rotation=pose.get("pelvis_rotation"),
            skeleton_only=True,
            rotation_type=model.rotation_type,
        )
        return _smpl_lbs_transforms(model.forward_skeleton(**pose), joints)
    if model_name == "smplx":
        _, joints, _, _ = smplx_core._forward_core(
            xp=np,
            v_template=None,
            shapedirs=None,
            exprdirs=None,
            j_template=model.weights.j_template,
            j_shapedirs=model.weights.j_shapedirs,
            j_exprdirs=model.weights.j_exprdirs,
            parents=model.weights.parents,
            kinematic_fronts=model.weights.kinematic_fronts,
            hand_mean=model.weights.hand_mean,
            shape=pose["shape"],
            expression=pose["expression"],
            body_pose=pose["body_pose"],
            hand_pose=pose["hand_pose"],
            head_pose=pose["head_pose"],
            pelvis_rotation=pose.get("pelvis_rotation"),
            skeleton_only=True,
            rotation_type=model.rotation_type,
        )
        return _smpl_lbs_transforms(model.forward_skeleton(**pose), joints)
    if model_name == "mhr":
        return _mhr_bone_transforms(model, pose)
    if model_name == "soma":
        return _soma_bone_transforms(model, pose)
    raise ValueError(f"Unsupported body model {model.__class__.__name__!r}.")


def _anny_pose(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Float[np.ndarray, "joints 3"]:
    return anny_pose.pack_pose(
        np,
        pose["global_rotation"],
        pose["body_pose"],
        pose["head_pose"],
        pose["hand_pose"],
    )


def _anny_unskinned(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> tuple[Float[np.ndarray, "vertices 3"], Float[np.ndarray, "joints 4 4"]]:
    weights = model.weights
    return anny_core.forward_unskinned_vertices(
        template_vertices=weights.template_vertices,
        blendshapes=weights.blendshapes,
        template_bone_heads=weights.template_bone_heads,
        template_bone_tails=weights.template_bone_tails,
        bone_heads_blendshapes=weights.bone_heads_blendshapes,
        bone_tails_blendshapes=weights.bone_tails_blendshapes,
        bone_rolls_rotmat=weights.bone_rolls_rotmat,
        phenotype_mask=weights.phenotype_mask,
        anchors=weights.anchors,
        kinematic_fronts=weights.kinematic_fronts,
        y_axis=weights.y_axis,
        degenerate_rotation=weights.degenerate_rotation,
        extrapolate_phenotypes=model.extrapolate_phenotypes,
        gender=pose["gender"],
        age=pose["age"],
        muscle=pose["muscle"],
        weight=pose["weight"],
        height=pose["height"],
        proportions=pose["proportions"],
        pose=_anny_pose(model, pose),
        rotation_type=model.rotation_type,
        xp=np,
    )


def _anny_base_vertices(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Float[np.ndarray, "vertices 3"]:
    vertices, _ = _anny_unskinned(model, pose)
    return _as_unbatched_array(vertices)


def _anny_bone_transforms(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Float[np.ndarray, "joints 4 4"]:
    _, transforms = _anny_unskinned(model, pose)
    return _apply_global_transform(transforms, None, pose.get("global_translation"))


def _smpl_base_vertices(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Float[np.ndarray, "vertices 3"]:
    shape = _as_unbatched_array(pose["shape"])
    return model.weights.v_template + np.einsum("s,vcs->vc", shape, model.weights.shapedirs[..., : shape.shape[-1]])


def _smplx_base_vertices(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Float[np.ndarray, "vertices 3"]:
    shape = _as_unbatched_array(pose["shape"])
    expression = _as_unbatched_array(pose["expression"])
    coeffs = np.concatenate([shape, expression])
    dirs = np.concatenate(
        [
            model.weights.shapedirs[..., : shape.shape[-1]],
            model.weights.exprdirs[..., : expression.shape[-1]],
        ],
        axis=-1,
    )
    return model.weights.v_template + np.einsum("p,vcp->vc", coeffs, dirs)


def _smpl_vertices(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
    base_vertices: Float[np.ndarray, "vertices 3"],
) -> Float[np.ndarray, "vertices 3"]:
    _, _, pose_matrices, _ = smpl_core._forward_core(
        xp=np,
        v_template=None,
        shapedirs=None,
        j_template=model.weights.j_template,
        j_shapedirs=model.weights.j_shapedirs,
        parents=model.weights.parents,
        kinematic_fronts=model.weights.kinematic_fronts,
        shape=pose["shape"],
        body_pose=pose["body_pose"],
        pelvis_rotation=pose.get("pelvis_rotation"),
        skeleton_only=True,
        rotation_type=model.rotation_type,
    )
    pose_delta = (pose_matrices[..., 1:, :, :] - np.eye(3, dtype=pose_matrices.dtype)).reshape(-1)
    vertices = base_vertices + (pose_delta @ model.weights.posedirs).reshape(-1, 3)
    return _as_unbatched_array(vertices)


def _smplh_vertices(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
    base_vertices: Float[np.ndarray, "vertices 3"],
) -> Float[np.ndarray, "vertices 3"]:
    _, _, pose_matrices, _ = smplh_core._forward_core(
        xp=np,
        v_template=None,
        shapedirs=None,
        j_template=model.weights.j_template,
        j_shapedirs=model.weights.j_shapedirs,
        parents=model.weights.parents,
        kinematic_fronts=model.weights.kinematic_fronts,
        hand_mean=model.weights.hand_mean,
        shape=pose["shape"],
        body_pose=pose["body_pose"],
        hand_pose=pose["hand_pose"],
        pelvis_rotation=pose.get("pelvis_rotation"),
        skeleton_only=True,
        rotation_type=model.rotation_type,
    )
    pose_delta = (pose_matrices[..., 1:, :, :] - np.eye(3, dtype=pose_matrices.dtype)).reshape(-1)
    vertices = base_vertices + (pose_delta @ model.weights.posedirs).reshape(-1, 3)
    return _as_unbatched_array(vertices)


def _smplx_vertices(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
    base_vertices: Float[np.ndarray, "vertices 3"],
) -> Float[np.ndarray, "vertices 3"]:
    _, _, pose_matrices, _ = smplx_core._forward_core(
        xp=np,
        v_template=None,
        shapedirs=None,
        exprdirs=None,
        j_template=model.weights.j_template,
        j_shapedirs=model.weights.j_shapedirs,
        j_exprdirs=model.weights.j_exprdirs,
        parents=model.weights.parents,
        kinematic_fronts=model.weights.kinematic_fronts,
        hand_mean=model.weights.hand_mean,
        shape=pose["shape"],
        expression=pose["expression"],
        body_pose=pose["body_pose"],
        hand_pose=pose["hand_pose"],
        head_pose=pose["head_pose"],
        pelvis_rotation=pose.get("pelvis_rotation"),
        skeleton_only=True,
        rotation_type=model.rotation_type,
    )
    pose_delta = (pose_matrices[..., 1:, :, :] - np.eye(3, dtype=pose_matrices.dtype)).reshape(-1)
    vertices = base_vertices + (pose_delta @ model.weights.posedirs).reshape(-1, 3)
    return _as_unbatched_array(vertices)


def _smpl_lbs_transforms(
    skeleton: Float[np.ndarray, "joints 4 4"] | Float[np.ndarray, "batch joints 4 4"],
    joints: Float[np.ndarray, "joints 3"] | Float[np.ndarray, "batch joints 3"],
) -> Float[np.ndarray, "joints 4 4"]:
    skeleton = _as_unbatched_array(skeleton)
    joints = _as_unbatched_array(joints)
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None], skeleton.shape[0], axis=0)
    transforms[:, :3, :3] = skeleton[:, :3, :3]
    transforms[:, :3, 3] = skeleton[:, :3, 3] - np.einsum("jkl,jl->jk", skeleton[:, :3, :3], joints)
    return transforms


def _mhr_vertices(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
    base_vertices: Float[np.ndarray, "vertices 3"],
) -> Float[np.ndarray, "vertices 3"]:
    packed_pose = pack_pose(np, pose["body_pose"], pose["hand_pose"])
    _, _, _, joint_params = _mhr_skeleton_core(model, packed_pose)
    vertices = base_vertices + mhr_core.apply_pose_correctives(
        joint_params,
        model.weights.corrective_W1,
        model.weights.corrective_W2,
        xp=np,
    )
    return _as_unbatched_array(vertices)


def _mhr_bone_transforms(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Float[np.ndarray, "joints 4 4"]:
    weights = model.weights
    packed_pose = pack_pose(np, pose["body_pose"], pose["hand_pose"])
    t_g, r_g, s_g, _ = _mhr_skeleton_core(model, packed_pose)
    lin_g = r_g * s_g[..., None]
    transforms = np.repeat(np.eye(4, dtype=np.float32)[None, None], model.num_joints, axis=1)
    transforms[..., :3, :3] = 0.01 * np.einsum("...jik,jkl->...jil", lin_g, weights.bind_inv_linear)
    transforms[..., :3, 3] = 0.01 * (
        np.einsum("...jik,jk->...ji", lin_g, weights.bind_inv_translation) + t_g
    )
    transforms = _apply_global_transform(transforms, pose.get("global_rotation"), pose.get("global_translation"))
    return _as_unbatched_array(transforms)


def _mhr_skeleton_core(
    model: Any,
    packed_pose: Float[np.ndarray, "joints parameters"],
) -> tuple[
    Float[np.ndarray, "batch joints 3"],
    Float[np.ndarray, "batch joints 3 3"],
    Float[np.ndarray, "batch joints 3"],
    Float[np.ndarray, "batch joints parameters"],
]:
    return mhr_core._forward_skeleton_core(
        xp=np,
        pose=packed_pose,
        joint_offsets=model.weights.joint_offsets,
        joint_pre_rotations=model.weights.joint_pre_rotations,
        parameter_transform=model.weights.parameter_transform,
        kinematic_fronts=model.weights.kinematic_fronts,
        num_joints=model.num_joints,
        shape_dim=model.SHAPE_DIM,
    )


def _soma_pose(
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Float[np.ndarray, "joints 3"]:
    return soma_pose.pack_pose(
        np,
        pose["global_rotation"],
        pose["body_pose"],
        pose["head_pose"],
        pose["hand_pose"],
    )


def _soma_prepared_identity(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Any:
    return model.prepare_identity(
        identity=pose.get("identity"),
        scale_params=pose.get("scale_params"),
        pose=_soma_pose(pose),
        cache=model.cache_identity,
    )


def _soma_pose_rot_full(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Float[np.ndarray, "joints 3 3"] | Float[np.ndarray, "batch joints 3 3"]:
    pose_rot = SO3.convert(_soma_pose(pose), src=model.rotation_type, dst="rotmat", xp=np)
    return soma_core._orient_pose_rot_full(
        np,
        pose_rot,
        model.weights.t_pose_world,
        model.weights.topology.parents_full,
    )


def _soma_vertices(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
    base_vertices: Float[np.ndarray, "vertices 3"],
) -> Float[np.ndarray, "vertices 3"]:
    correctives = model._kernel.apply_pose_correctives(model.weights, _soma_pose_rot_full(model, pose), xp=np)
    if model.weights.vertex_map is not None:
        correctives = correctives[..., model.weights.vertex_map, :]
    return base_vertices + _as_unbatched_array(correctives) * 0.01


def _soma_bone_transforms(
    model: Any,
    pose: dict[
        str,
        Float[np.ndarray, "dim"]
        | Float[np.ndarray, "items dim"]
        | Float[np.ndarray, "items rows cols"],
    ],
) -> Float[np.ndarray, "joints 4 4"]:
    prepared = _soma_prepared_identity(model, pose)
    world_bind_pose = prepared.world_bind_pose
    pose_rot_full = _soma_pose_rot_full(model, pose)
    world = soma_core._pose_skeleton_from_oriented_pose(
        xp=np,
        world_bind_pose=world_bind_pose,
        kinematic_fronts=model.weights.topology.kinematic_fronts_full,
        parents_full=model.weights.topology.parents_full,
        pose_rot_full=pose_rot_full,
    )
    transforms = world @ prepared.inverse_world_bind_pose
    transforms = _as_unbatched_array(transforms)[1:]
    transforms[:, :3, 3] *= 0.01
    return _apply_global_transform(transforms, pose.get("global_rotation"), pose.get("global_translation"))


def _apply_global_transform(
    transforms: Float[np.ndarray, "joints 4 4"] | Float[np.ndarray, "batch joints 4 4"],
    rotation: Float[np.ndarray, "3"] | Float[np.ndarray, "batch 3"] | None,
    translation: Float[np.ndarray, "3"] | Float[np.ndarray, "batch 3"] | None,
) -> Float[np.ndarray, "joints 4 4"] | Float[np.ndarray, "batch joints 4 4"]:
    if rotation is None and translation is None:
        return transforms

    global_transform = np.eye(4, dtype=np.float32)
    if rotation is not None:
        global_transform[:3, :3] = _as_unbatched_array(SO3.conversions.from_axis_angle_to_rotmat(rotation, xp=np))
    if translation is not None:
        global_transform[:3, 3] = _as_unbatched_array(translation)
    return np.einsum("ij,...jk->...ik", global_transform, transforms)


def _as_unbatched_array(
    value: Any,
) -> Float[np.ndarray, "dim"] | Float[np.ndarray, "items dim"] | Float[np.ndarray, "items rows cols"]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    return value[0] if value.ndim >= 2 and value.shape[0] == 1 else value


def _triangular_faces(model: Any) -> Int[np.ndarray, "faces 3"]:
    faces = np.asarray(model.faces, dtype=np.uint32)
    if faces.shape[1] == 3:
        return faces
    if faces.shape[1] == 4:
        return np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]], axis=0)
    raise ValueError(f"Expected triangular or quad faces, got {faces.shape}.")


def _sparse_skinning(
    model: Any,
) -> tuple[Float[np.ndarray, "vertices influences"], Int[np.ndarray, "vertices influences"]]:
    model_name = model.__class__.__name__.lower()
    if model_name == "soma":
        indices = model.weights.skin_joint_indices_active.copy()
        indices[indices >= 0] -= 1
        return _compact_skinning(model.weights.skin_joint_weights_active, indices)

    if hasattr(model.weights, "skin_indices") and hasattr(model.weights, "skin_weights"):
        return model.weights.skin_weights, model.weights.skin_indices

    dense = np.asarray(model.skin_weights)
    return _compact_skinning(dense, np.broadcast_to(np.arange(dense.shape[1]), dense.shape))


def _compact_skinning(
    weights: Float[np.ndarray, "vertices source_influences"],
    joints: Int[np.ndarray, "vertices source_influences"],
) -> tuple[Float[np.ndarray, "vertices influences"], Int[np.ndarray, "vertices influences"]]:
    skin_joints: list[Int[np.ndarray, "influences"]] = []
    skin_weights: list[Float[np.ndarray, "influences"]] = []
    for weight_row, joint_row in zip(weights, joints):
        active = np.flatnonzero(weight_row)
        skin_weights.append(weight_row[active])
        skin_joints.append(joint_row[active])

    max_len = max(len(row) for row in skin_joints)
    joints_out = np.zeros((len(skin_joints), max_len), dtype=np.int32)
    weights_out = np.zeros((len(skin_weights), max_len), dtype=np.float32)
    for vertex, (joints, weights) in enumerate(zip(skin_joints, skin_weights)):
        joints_out[vertex, : len(joints)] = joints
        weights_out[vertex, : len(weights)] = weights
    return weights_out, joints_out
