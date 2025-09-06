import numpy as np
import tkenginer.math as math


def test_projection_matrix():
    fov, width, height, near, far = 90.0, 800, 600, 0.1, 100.0
    proj = math.get_projection_matrix(fov, width, height, near, far)
    assert proj.shape == (4, 4)
    assert np.isclose(proj[0, 0], proj[1, 1] * (height / width))


def test_camera_vectors():
    yaw = np.radians(90.0)
    pitch = 0.0
    front, right, up = math.get_camera_vectors(yaw, pitch)
    assert np.allclose(np.linalg.norm(front), 1.0)
    assert np.allclose(np.dot(front, right), 0.0)
    assert np.allclose(np.dot(front, up), 0.0)
    assert np.allclose(np.dot(right, up), 0.0)


def test_view_matrix():
    pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    yaw, pitch = 0.0, 0.0
    view = math.get_view_matrix(pos, yaw, pitch)
    assert view.shape == (4, 4)
    assert np.allclose(view[3, :], [0, 0, 0, 1])


def test_transform_vertices():
    vertices = np.array([[1, 2, 3]], dtype=np.float32)
    mvp = np.eye(4, dtype=np.float32)
    out = math.transform_vertices(vertices, mvp)
    assert out.shape == (1, 4)
    assert np.allclose(out[0], [1, 2, 3, 1])


def test_clip_to_screen():
    clip_coords = np.array([[0, 0, 0, 1]], dtype=np.float32)
    width, height = 100, 100
    screen, w = math.clip_to_screen(clip_coords, width, height)
    assert screen.shape == (1, 2)
    assert np.all(screen[0] == [50, 50])
    assert np.all(w == 1)


def test_lerp():
    assert math.lerp(0.0, 10.0, 0.0) == 0.0
    assert math.lerp(0.0, 10.0, 1.0) == 10.0
    assert math.lerp(0.0, 10.0, 0.5) == 5.0


def test_barycentric_inside_triangle():
    p0 = (0.0, 0.0)
    p1 = (2.0, 0.0)
    p2 = (0.0, 2.0)
    u, v, w = math.barycentric_weights(0.5, 0.5, p0, p1, p2)
    assert np.isclose(u + v + w, 1.0)
    assert u >= 0 and v >= 0 and w >= 0


def test_barycentric_outside_triangle():
    p0 = (0.0, 0.0)
    p1 = (2.0, 0.0)
    p2 = (0.0, 2.0)
    u, v, w = math.barycentric_weights(3.0, 3.0, p0, p1, p2)
    assert u < 0 or v < 0 or w < 0
