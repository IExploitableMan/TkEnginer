import numpy as np
import io


class Mesh:
    def __init__(
        self,
        vertices: list[list[float]],
        indices: list[list[int]],
        colors: list[list[int]] = None
    ) -> None:
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        if colors is not None:
            self.colors = np.array(colors, dtype=np.uint8)
        else:
            self.colors = np.full((len(vertices), 4), 255, dtype=np.uint8)

    def get_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.vertices, self.indices, self.colors


class CubeMesh(Mesh):
    def __init__(self, colors: list[list[int]] = None) -> None:
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
            [5, 4, 7], [5, 7, 6],
            [4, 0, 3], [4, 3, 7],
            [1, 5, 6], [1, 6, 2],
            [3, 2, 6], [3, 6, 7],
            [4, 5, 1], [4, 1, 0],
        ]
        super().__init__(vertices, indices, colors)


class PyramidMesh(Mesh):
    def __init__(self, colors: list[list[int]] = None) -> None:
        vertices = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.0, -0.5, 0.5],
            [0.0, 0.5, 0.0],
        ]
        indices = [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ]
        super().__init__(vertices, indices, colors)


class PyramidWithSquareBaseMesh(Mesh):
    def __init__(self, colors: list[list[int]] = None) -> None:
        vertices = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, -0.5, 0.5],
            [0.0, 0.5, 0.0],
        ]
        indices = [
            [0, 3, 2],
            [0, 2, 1],
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ]
        super().__init__(vertices, indices, colors)


class SphereMesh(Mesh):
    def __init__(self, segments: int, colors: list[list[int]] = None) -> None:
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
        super().__init__(vertices, indices, colors)


class ConeMesh(Mesh):
    def __init__(self, segments: int, colors: list[list[int]] = None) -> None:
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

        super().__init__(vertices, indices, colors)


class CylinderMesh(Mesh):
    def __init__(self, segments: int, colors: list[list[int]] = None) -> None:
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

        vertices = [top_center] + top_vertices + \
            [bottom_center] + bottom_vertices

        indices = []
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.append([0, i + 1, next_i + 1])

        bottom_center_idx = len(top_vertices) + 1
        bottom_vertex_start = bottom_center_idx + 1
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.append(
                [bottom_center_idx, bottom_vertex_start + next_i, bottom_vertex_start + i])

        for i in range(segments):
            next_i = (i + 1) % segments
            top_idx = i + 1
            next_top_idx = next_i + 1
            bottom_idx = bottom_vertex_start + i
            next_bottom_idx = bottom_vertex_start + next_i

            indices.append([top_idx, bottom_idx, next_top_idx])
            indices.append([next_top_idx, bottom_idx, next_bottom_idx])

        super().__init__(vertices, indices, colors)


class PlaneMesh(Mesh):
    def __init__(self, colors: list[list[int]] = None) -> None:
        vertices = [
            [-0.5, 0.0, -0.5],
            [0.5, 0.0, -0.5],
            [0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
        ]
        indices = [
            [0, 1, 2],
            [0, 2, 3],
        ]
        super().__init__(vertices, indices, colors)


class OBJMesh(Mesh):
    def __init__(self, file: io.TextIOWrapper, colors: list[list[int]] = None) -> None:
        vertices = []
        indices = []

        with file:  # TODO: implement MTL parser, textures
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    vertices.append([float(coord) for coord in parts[1:4]])
                elif parts[0] == "f":
                    buffer = [
                        int(part.split("/")[0])
                        -
                        1 for part in parts[1:]
                    ]
                    if len(buffer) == 3:
                        indices.append(buffer)
                    elif len(buffer) > 3:
                        for i in range(1, len(buffer) - 1):
                            indices.append(
                                [buffer[0], buffer[i], buffer[i + 1]])

        super().__init__(vertices, indices, colors)
