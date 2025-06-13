import numpy as np


def get_projection_matrix(fov: float, width: int, height: int, near: float, far: float) -> np.ndarray:
    focal = 1 / np.tan(np.radians(fov) / 2)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = focal / (width / height)
    proj[1, 1] = focal
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1
    return proj


def get_camera_vectors(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    front = np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw)
    ], dtype=np.float32)
    front /= np.linalg.norm(front)

    right = np.cross([0, 1, 0], front)
    right /= np.linalg.norm(right)

    up = np.cross(front, right)

    return front, right, up


def get_view_matrix(position: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    front, right, up = get_camera_vectors(yaw, pitch)

    view = np.identity(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -front
    view[:3, 3] = -view[:3, :3] @ position
    return view


def transform_vertices(vertices: np.ndarray, mvp_matrix: np.ndarray) -> np.ndarray:
    vertices_hom = np.hstack(
        [vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)])
    vertices_clip = vertices_hom @ mvp_matrix.T
    return vertices_clip


def clip_to_screen(vertices_clip: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    w_coords = vertices_clip[:, 3:4]
    vertices_ndc = vertices_clip[:, :3] / w_coords
    screen_coords = np.empty((len(vertices_ndc), 2), dtype=np.int32)
    screen_coords[:, 0] = ((vertices_ndc[:, 0] + 1) *
                           0.5 * width).astype(np.int32)
    screen_coords[:, 1] = (
        (1 - (vertices_ndc[:, 1] + 1) * 0.5) * height).astype(np.int32)
    return screen_coords, w_coords


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# def lerp_color(c0, c1, t):
#     return tuple(int(lerp(c0[i], c1[i], t)) for i in range(3))


# def barycentric_interp_color(c0, c1, c2, u, v):
#     w = 1 - u - v
#     return tuple(int(c0[i]*w + c1[i]*u + c2[i]*v) for i in range(3))
