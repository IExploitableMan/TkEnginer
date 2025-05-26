import numpy as np


class Transform:
    def __init__(
        self,
        position: list[float] = None,
        rotation: list[float] = None,
        scale: list[float] = None
    ) -> None:
        if position is None:
            position = [0.0, 0.0, 0.0]
        if rotation is None:
            rotation = [0.0, 0.0, 0.0]
        if scale is None:
            scale = [1.0, 1.0, 1.0]
        self.position = np.array(position, dtype=np.float32)
        self.rotation = np.array(rotation, dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)

    def get_matrix(self) -> np.typing.NDArray[np.float32]:
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
