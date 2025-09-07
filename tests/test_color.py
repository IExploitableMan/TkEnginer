"""
Tests for the color module.
"""

import numpy as np

from tkenginer.color import *

def test_color_creation():
    """
    Tests the creation of a Color object.
    """
    c = Color(10, 20, 30, 40)
    assert c.r == 10
    assert c.g == 20
    assert c.b == 30
    assert c.a == 40

def test_color_creation_default_alpha():
    """
    Tests the creation of a Color object with a default alpha value.
    """
    c = Color(10, 20, 30)
    assert c.r == 10
    assert c.g == 20
    assert c.b == 30
    assert c.a == 255

def test_color_to_tuple():
    """
    Tests the conversion of a Color object to a tuple.
    """
    c = Color(10, 20, 30, 40)
    assert c.to_tuple() == (10, 20, 30, 40)

def test_color_to_numpy():
    """
    Tests the conversion of a Color object to a NumPy array.
    """
    c = Color(10, 20, 30, 40)
    np.testing.assert_array_equal(c.to_numpy(), np.array([10, 20, 30, 40], dtype=np.uint8))

def test_color_from_tuple_rgb():
    """
    Tests the creation of a Color object from an RGB tuple.
    """
    c = Color.from_tuple((10, 20, 30))
    assert c.r == 10
    assert c.g == 20
    assert c.b == 30
    assert c.a == 255

def test_color_from_tuple_rgba():
    """
    Tests the creation of a Color object from an RGBA tuple.
    """
    c = Color.from_tuple((10, 20, 30, 40))
    assert c.r == 10
    assert c.g == 20
    assert c.b == 30
    assert c.a == 40
