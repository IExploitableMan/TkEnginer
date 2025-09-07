"""
This module provides color-related classes and constants.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Color:
    """
    Represents a color with red, green, blue, and alpha components.
    """
    r: int
    g: int
    b: int
    a: int = 255

    def to_tuple(self) -> tuple[int, int, int, int]:
        """
        Converts the color to a tuple of (r, g, b, a).

        Returns:
            A tuple representing the color.
        """
        return (self.r, self.g, self.b, self.a)

    def to_numpy(self) -> np.ndarray:
        """
        Converts the color to a NumPy array.

        Returns:
            A NumPy array representing the color.
        """
        return np.array([self.r, self.g, self.b, self.a], dtype=np.uint8)

    @staticmethod
    def from_tuple(rgba: tuple[int, int, int] | tuple[int, int, int, int]) -> "Color":
        """
        Creates a Color object from a tuple.

        Args:
            rgba: A tuple of (r, g, b) or (r, g, b, a).

        Returns:
            A Color object.
        """
        if len(rgba) == 3:
            return Color(rgba[0], rgba[1], rgba[2])
        return Color(rgba[0], rgba[1], rgba[2], rgba[3])


class Colors:
    """
    A collection of predefined colors.
    """
    BLACK = Color(0, 0, 0)
    WHITE = Color(255, 255, 255)
    RED = Color(255, 0, 0)
    GREEN = Color(0, 255, 0)
    BLUE = Color(0, 0, 255)
    YELLOW = Color(255, 255, 0)
