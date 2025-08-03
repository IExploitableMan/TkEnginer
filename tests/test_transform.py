import numpy as np
import pytest
from tkenginer.transform import Transform


def assert_matrix_equal(a, b):
    assert np.allclose(a, b, atol=1e-6)


def test_default_get_matrix():
    t = Transform()
    m = t.get_matrix()
    assert isinstance(m, np.ndarray)
    assert m.shape == (4, 4)
    assert m.dtype == np.float32
    assert_matrix_equal(m, np.identity(4, dtype=np.float32))


def test_translation():
    t = Transform(position=[1, 2, 3])
    m = t.get_matrix()
    assert np.allclose(m[:3, 3], [1, 2, 3], atol=1e-6)


def test_scaling():
    t = Transform(scale=[2, 3, 4])
    m = t.get_matrix()
    assert np.allclose(np.diag(m)[:3], [2, 3, 4], atol=1e-6)


def test_rotation_z_90():
    angle = np.pi / 2
    t = Transform(rotation=[0, 0, angle])
    m = t.get_matrix()
    expected = np.array([
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ], dtype=np.float32)
    assert_matrix_equal(m, expected)


def test_from_matrix_roundtrip():
    pos = [1.1, -2.2, 3.3]
    rot = [0.1, 0.2, -0.3]
    scale = [2.0, 3.0, 4.0]
    t1 = Transform(position=pos, rotation=rot, scale=scale)
    m = t1.get_matrix()
    t2 = Transform.from_matrix(m)
    assert np.allclose(t1.position, t2.position, atol=1e-6)
    assert np.allclose(t1.rotation, t2.rotation, atol=1e-6)
    assert np.allclose(t1.scale,    t2.scale,    atol=1e-6)


def test_matmul_combines_transforms():
    t1 = Transform(position=[1, 0, 0])
    t2 = Transform(position=[0, 1, 0])
    t3 = t1 @ t2
    m3 = t3.get_matrix()
    expected = Transform(position=[1, 1, 0]).get_matrix()
    assert_matrix_equal(m3, expected)


def test_non_commutative():
    t1 = Transform(rotation=[0, 0, np.pi/4])
    t2 = Transform(rotation=[0, np.pi/6, 0])
    m12 = (t1 @ t2).get_matrix()
    m21 = (t2 @ t1).get_matrix()
    assert not np.allclose(m12, m21, atol=1e-6)


def test_invalid_init_lengths():
    with pytest.raises(Exception):
        Transform(position=[1, 2]).get_matrix()
    with pytest.raises(Exception):
        Transform(rotation=[0, 1]).get_matrix()
    with pytest.raises(Exception):
        Transform(scale=[1, 2]).get_matrix()


def test_gimbal_lock_cases():
    y1 = -np.pi/2
    t1 = Transform(rotation=[0.0, y1, 0.0])
    m1 = t1.get_matrix()
    t1r = Transform.from_matrix(m1)
    assert_matrix_equal(m1, t1r.get_matrix())

    y2 = np.pi/2
    t2 = Transform(rotation=[0.0, y2, 0.0])
    m2 = t2.get_matrix()
    t2r = Transform.from_matrix(m2)
    assert_matrix_equal(m2, t2r.get_matrix())
