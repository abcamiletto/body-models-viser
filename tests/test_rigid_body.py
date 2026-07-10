from __future__ import annotations

import numpy as np
import pytest
import trimesh
from body_models import RigidBodyModel

import body_models_viser as bmv

LOCAL_VERTICES = (
    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.5]]),
)


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class StubRigidModel(RigidBodyModel):
    """Two one-triangle links; link i rotates about z by body_pose[i] at x=i."""

    parents = [-1, 0]
    link_vertex_starts = [0, 3]
    link_vertex_counts = [3, 3]
    link_face_starts = [0, 1]
    link_face_counts = [1, 1]

    @property
    def faces(self):
        return np.array([[0, 1, 2], [3, 4, 5]])

    @property
    def num_vertices(self):
        return 6

    @property
    def joint_names(self):
        return ["root", "child"]

    @property
    def actuated_joint_names(self):
        return ["root", "child"]

    @property
    def actuated_joint_limits(self):
        return np.array([[-np.pi, np.pi], [-np.pi, np.pi]])

    @property
    def actuated_joint_types(self):
        return ["hinge", "hinge"]

    @property
    def link_names(self):
        return ["link0", "link1"]

    @property
    def link_joint_indices(self):
        return [0, 1]

    def get_rest_pose(self, batch_dims=()):
        return {
            "body_pose": np.array([0.4, -0.7], dtype=np.float32),
            "global_translation": np.zeros(3, dtype=np.float32),
        }

    def forward_links(self, body_pose, global_translation):
        transforms = []
        for index, angle in enumerate(np.asarray(body_pose)):
            transform = np.eye(4)
            transform[:3, :3] = rotation_z(float(angle))
            transform[:3, 3] = [float(index), 0.0, 0.0] + np.asarray(global_translation)
            transforms.append(transform)
        return np.stack(transforms)

    def forward_skeleton(self, body_pose, global_translation):
        return self.forward_links(body_pose, global_translation)

    def forward_meshes(self, body_pose, global_translation):
        links = self.forward_links(body_pose, global_translation)
        world = [
            vertices @ links[index, :3, :3].T + links[index, :3, 3]
            for index, vertices in enumerate(LOCAL_VERTICES)
        ]
        return [trimesh.Trimesh(np.concatenate(world), self.faces, process=False)]


def test_add_bakes_link_local_meshes(scene, monkeypatch):
    recorded = []
    original = scene.add_mesh_simple

    def spy(name, *, vertices, faces, **kwargs):
        recorded.append((vertices, faces))
        return original(name, vertices=vertices, faces=faces, **kwargs)

    monkeypatch.setattr(scene, "add_mesh_simple", spy)
    handle = bmv.add_rigid_body_model(scene, "/rigid_add", StubRigidModel())

    assert len(handle.links) == 2
    for (vertices, faces), local in zip(recorded, LOCAL_VERTICES):
        np.testing.assert_allclose(vertices, local, atol=1e-6)
        np.testing.assert_array_equal(faces, [[0, 1, 2]])
    handle.remove()


def test_add_places_links_at_rest_transforms(scene):
    model = StubRigidModel()
    handle = bmv.add_rigid_body_model(scene, "/rigid_rest", model)

    rest_links = model.forward_links(**model.get_rest_pose())
    for link, transform in zip(handle.links, rest_links):
        np.testing.assert_allclose(link.position, transform[:3, 3], atol=1e-6)
    handle.remove()


def test_set_pose_moves_links(scene):
    handle = bmv.add_rigid_body_model(scene, "/rigid_pose", StubRigidModel())
    posed = np.array([0.0, np.pi / 2], dtype=np.float32)

    handle.set_pose(body_pose=posed)

    np.testing.assert_allclose(handle.links[1].position, [1.0, 0.0, 0.0], atol=1e-6)
    half = np.pi / 4
    np.testing.assert_allclose(handle.links[1].wxyz, [np.cos(half), 0.0, 0.0, np.sin(half)], atol=1e-6)
    np.testing.assert_array_equal(handle.body_pose, posed)
    handle.remove()


def test_invalid_pose_key_raises(scene):
    handle = bmv.add_rigid_body_model(scene, "/rigid_invalid", StubRigidModel())
    with pytest.raises(ValueError):
        handle.set_pose(bogus=np.zeros(2))
    handle.remove()


def test_non_rigid_model_raises(scene):
    with pytest.raises(TypeError):
        bmv.add_rigid_body_model(scene, "/rigid_nope", object())
