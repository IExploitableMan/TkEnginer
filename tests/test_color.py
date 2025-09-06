import numpy as np

from tkenginer.color import *

def test_color_creation():
    c = Color(10, 20, 30, 40)
    assert c.r == 10
    assert c.g == 20
    assert c.b == 30
    assert c.a == 40

def test_color_creation_default_alpha():
    c = Color(10, 20, 30)
    assert c.r == 10
    assert c.g == 20
    assert c.b == 30
    assert c.a == 255

def test_color_to_tuple():
    c = Color(10, 20, 30, 40)
    assert c.to_tuple() == (10, 20, 30, 40)

def test_color_to_numpy():
    c = Color(10, 20, 30, 40)
    np.testing.assert_array_equal(c.to_numpy(), np.array([10, 20, 30, 40], dtype=np.uint8))

def test_color_from_tuple_rgb():
    c = Color.from_tuple((10, 20, 30))
    assert c.r == 10
    assert c.g == 20
    assert c.b == 30
    assert c.a == 255

def test_color_from_tuple_rgba():
    c = Color.from_tuple((10, 20, 30, 40))
    assert c.r == 10
    assert c.g == 20
    assert c.b == 30
    assert c.a == 40
