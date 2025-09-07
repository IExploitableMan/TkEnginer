"""
This module provides the Transform class for representing 3D transformations.
"""

import numpy as np


class Transform:
    """
    Represents a 3D transformation with position, rotation, and scale.
    """

    def __init__(
        self,
        position: list[float] = None,
        rotation: list[float] = None,
        scale: list[float] = None
    ) -> None:
        """
        Initializes the transform.

        Args:
            position: The position as a list of 3 floats (x, y, z).
            rotation: The rotation as a list of 3 floats (pitch, yaw, roll) in radians.
            scale: The scale as a list of 3 floats (x, y, z).
        """
        position = position if position is not None else [0.0, 0.0, 0.0]
        rotation = rotation if rotation is not None else [0.0, 0.0, 0.0]
        scale = scale if scale is not None else [1.0, 1.0, 1.0]
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)

    def get_matrix(self) -> np.typing.NDArray[np.float32]:
        """
        Returns the transformation matrix.

        Returns:
            The 4x4 transformation matrix.
        """
        translation_matrix = np.identity(4, dtype=np.float32)
        translation_matrix[0, 3] = self.position[0]
        translation_matrix[1, 3] = self.position[1]
        translation_matrix[2, 3] = self.position[2]

        rotation_matrix = np.array([
            [np.cos(self.rotation[2]), -np.sin(self.rotation[2]), 0, 0],
            [np.sin(self.rotation[2]),  np.cos(self.rotation[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32) @ np.array([
            [np.cos(self.rotation[1]), 0, np.sin(self.rotation[1]), 0],
            [0, 1, 0, 0],
            [-np.sin(self.rotation[1]), 0, np.cos(self.rotation[1]), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32) @ np.array([
            [1, 0, 0, 0],
            [0, np.cos(self.rotation[0]), -np.sin(self.rotation[0]), 0],
            [0, np.sin(self.rotation[0]),  np.cos(self.rotation[0]), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        scaling_matrix = np.identity(4, dtype=np.float32)
        scaling_matrix[0, 0] = self.scale[0]
        scaling_matrix[1, 1] = self.scale[1]
        scaling_matrix[2, 2] = self.scale[2]

        return translation_matrix @ rotation_matrix @ scaling_matrix

    @classmethod
    def from_matrix(cls, matrix: np.typing.NDArray[np.float32]) -> "Transform":
        """
        Creates a Transform object from a transformation matrix.

        Args:
            matrix: The 4x4 transformation matrix.

        Returns:
            A new Transform object.
        """
        scale = np.linalg.norm(matrix[:3, :3], axis=0)

        rotation_matrix = matrix[:3, :3] / scale
        if np.abs(rotation_matrix[2, 0]) != 1:
            y = -np.arcsin(rotation_matrix[2, 0])
            cos_y = np.cos(y)
            x = np.arctan2(rotation_matrix[2, 1] /
                           cos_y, rotation_matrix[2, 2] / cos_y)
            z = np.arctan2(rotation_matrix[1, 0] /
                           cos_y, rotation_matrix[0, 0] / cos_y)
        else:
            z = 0
            if rotation_matrix[2, 0] == -1:
                y = np.pi / 2
                x = z + \
                    np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 2])
            else:
                y = -np.pi / 2
                x = -z + \
                    np.arctan2(-rotation_matrix[0, 1], -rotation_matrix[0, 2])
        rotation = np.array([x, y, z], dtype=np.float32)

        return cls(matrix[:3, 3], rotation, scale)

    def __matmul__(self, other: "Transform") -> np.ndarray:
        """
        Combines two transforms.

        Args:
            other: The other transform to combine with.

        Returns:
            A new Transform object representing the combined transform.
        """
        return Transform.from_matrix(self.get_matrix() @ other.get_matrix())
    
    def __eq__(self, value):
        """
        Checks if two transforms are equal.

        Args:
            value: The other object to compare to.

        Returns:
            True if the transforms are equal, False otherwise.
        """
        if not isinstance(value, Transform):
            return False
        return (np.allclose(self.position, value.position) and
                np.allclose(self.rotation, value.rotation) and
                np.allclose(self.scale, value.scale))
