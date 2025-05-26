import numpy as np


class Mesh:
    def __init__(self, vertices: list[list[float]], indices: list[list[int]]) -> None:
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices, self.indices


class CubeMesh(Mesh):
    def __init__(self) -> None:
        vertices = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5,  0.5, -0.5],
            [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5],
            [0.5, -0.5,  0.5],
            [0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5],
        ]
        indices = [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
            [3, 2, 6], [3, 6, 7],
            [0, 1, 5], [0, 5, 4],
        ]
        super().__init__(vertices=vertices, indices=indices)


class PyramidMesh(Mesh):
    def __init__(self) -> None:
        vertices = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.0, -0.5, 0.5],
            [0.0, 0.5, 0.0],
        ]
        indices = [
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ]
        super().__init__(vertices=vertices, indices=indices)


class PyramidWithSquareBaseMesh(Mesh):
    def __init__(self) -> None:
        vertices = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [0.0, 0.5, 0.0],
        ]
        indices = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ]
        super().__init__(vertices=vertices, indices=indices)


class SphereMesh(Mesh):
    def __init__(self, segments: int):
        vertices = []
        indices = []

        for i in range(segments + 1):
            latitude = np.pi * i / segments
            for j in range(segments + 1):
                longitude = 2 * np.pi * j / segments
                vertices.append([
                    np.sin(latitude) * np.cos(longitude),
                    np.cos(latitude),
                    np.sin(latitude) * np.sin(longitude),
                ])

        for i in range(segments):
            for j in range(segments):
                x = i * (segments + 1) + j
                y = x + segments + 1
                indices.extend([
                    [x, y, x + 1],
                    [y, y + 1, x + 1]
                ])
        super().__init__(vertices, indices)


class ConeMesh(Mesh):
    def __init__(self, segments):
        top_vertex = [0.0, 0.5, 0.0]
        base_vertices = []

        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = 0.5 * np.cos(angle)
            z = 0.5 * np.sin(angle)
            base_vertices.append([x, -0.5, z])

        base_center = [0.0, -0.5, 0.0]
        vertices = [top_vertex] + base_vertices + [base_center]

        indices = []
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.append([0, i + 1, next_i + 1])

        base_center_idx = len(vertices) - 1
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.append([base_center_idx, next_i + 1, i + 1])

        super().__init__(vertices, indices)


class CylinderMesh(Mesh):
    def __init__(self, segments):
        top_center = [0.0, 0.5, 0.0]
        bottom_center = [0.0, -0.5, 0.0]
        top_vertices = []
        bottom_vertices = []

        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = 0.5 * np.cos(angle)
            z = 0.5 * np.sin(angle)
            top_vertices.append([x, 0.5, z])
            bottom_vertices.append([x, -0.5, z])

        vertices = [top_center] + top_vertices + [bottom_center] + bottom_vertices

        indices = []
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.append([0, i + 1, next_i + 1])

        bottom_center_idx = len(top_vertices) + 1
        bottom_vertex_start = bottom_center_idx + 1
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.append([bottom_center_idx, bottom_vertex_start + next_i, bottom_vertex_start + i])

        for i in range(segments):
            next_i = (i + 1) % segments
            top_idx = i + 1
            next_top_idx = next_i + 1
            bottom_idx = bottom_vertex_start + i
            next_bottom_idx = bottom_vertex_start + next_i

            indices.append([top_idx, bottom_idx, next_top_idx])
            indices.append([next_top_idx, bottom_idx, next_bottom_idx])

        super().__init__(vertices, indices)

# TODO: OBJ loader
