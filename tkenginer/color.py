from dataclasses import dataclass
import numpy as np


@dataclass
class Color:
    r: int
    g: int
    b: int
    a: int = 255

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.r, self.g, self.b, self.a], dtype=np.uint8)

    @staticmethod
    def from_tuple(rgba: tuple[int, int, int] | tuple[int, int, int, int]) -> "Color":
        if len(rgba) == 3:
            return Color(rgba[0], rgba[1], rgba[2])
        return Color(rgba[0], rgba[1], rgba[2], rgba[3])

class Colors:
    BLACK = Color(0, 0, 0)
    WHITE = Color(255, 255, 255)
    RED = Color(255, 0, 0)
    GREEN = Color(0, 255, 0)
    BLUE = Color(0, 0, 255)
    YELLOW = Color(255, 255, 0)
