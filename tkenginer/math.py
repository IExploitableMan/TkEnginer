"""
This module provides mathematical functions for 3D graphics.
"""

import numpy as np
import numba as nb


@nb.njit(cache=True)
def get_projection_matrix(
    fov: float,
    width: int,
    height: int,
    near: float,
    far: float
) -> np.ndarray:
    """
    Creates a perspective projection matrix.

    Args:
        fov: The field of view in degrees.
        width: The width of the viewport.
        height: The height of the viewport.
        near: The near clipping plane.
        far: The far clipping plane.

    Returns:
        The projection matrix.
    """
    focal = 1 / np.tan(np.radians(fov) / 2)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = focal / (width / height)
    proj[1, 1] = focal
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1
    return proj


@nb.njit(cache=True)
def get_camera_vectors(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the front, right, and up vectors for the camera.

    Args:
        yaw: The yaw of the camera in radians.
        pitch: The pitch of the camera in radians.

    Returns:
        A tuple containing the front, right, and up vectors.
    """
    front = np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw)
    ], dtype=np.float32)
    front /= np.linalg.norm(front)

    right = np.cross(np.array([0, 1, 0], dtype=np.float32), front)
    right /= np.linalg.norm(right)

    up = np.cross(front, right)

    return front, right, up


@nb.njit(cache=True)
def get_view_matrix(position: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """
    Creates a view matrix.

    Args:
        position: The position of the camera.
        yaw: The yaw of the camera in radians.
        pitch: The pitch of the camera in radians.

    Returns:
        The view matrix.
    """
    front, right, up = get_camera_vectors(yaw, pitch)

    view = np.identity(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -front
    view[:3, 3] = -view[:3, :3] @ position
    return view


@nb.njit(cache=True)
def transform_vertex(vertex: np.ndarray, mvp_matrix: np.ndarray) -> np.ndarray:
    """
    Transforms a single vertex by a matrix.

    Args:
        vertex: The vertex to transform.
        mvp_matrix: The matrix to transform the vertex by.

    Returns:
        The transformed vertex.
    """
    vertex_hom = np.array(
        [vertex[0], vertex[1], vertex[2], 1.0], dtype=np.float32)
    vertex_clip = vertex_hom @ mvp_matrix.T
    return vertex_clip


@nb.njit(cache=True)
def transform_vertices(vertices: np.ndarray, mvp_matrix: np.ndarray) -> np.ndarray:
    """
    Transforms multiple vertices by a matrix.

    Args:
        vertices: The vertices to transform.
        mvp_matrix: The matrix to transform the vertices by.

    Returns:
        The transformed vertices.
    """
    vertices_hom = np.concatenate(
        (vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)),
        axis=1
    )
    vertices_clip = vertices_hom @ mvp_matrix.T
    return vertices_clip


@nb.njit(cache=True)
def clip_to_screen(vertices_clip: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts clip-space coordinates to screen coordinates.

    Args:
        vertices_clip: The vertices in clip space.
        width: The width of the viewport.
        height: The height of the viewport.

    Returns:
        A tuple containing the screen coordinates and the w-coordinates.
    """
    w_coords = vertices_clip[:, 3:4]
    vertices_ndc = vertices_clip[:, :3] / w_coords
    screen_coords = np.empty((len(vertices_ndc), 2), dtype=np.int32)
    screen_coords[:, 0] = ((vertices_ndc[:, 0] + 1) *
                           0.5 * width).astype(np.int32)
    screen_coords[:, 1] = (
        (1 - (vertices_ndc[:, 1] + 1) * 0.5) * height).astype(np.int32)
    return screen_coords, w_coords


@nb.njit(cache=True)
def lerp(a: float, b: float, t: float) -> float:
    """
    Performs linear interpolation.

    Args:
        a: The start value.
        b: The end value.
        t: The interpolation factor.

    Returns:
        The interpolated value.
    """
    return a + (b - a) * t


@nb.njit(cache=True)
def barycentric_weights(px, py, p0, p1, p2):
    """
    Calculates barycentric weights for a point in a triangle.

    Args:
        px: The x-coordinate of the point.
        py: The y-coordinate of the point.
        p0: The first vertex of the triangle.
        p1: The second vertex of the triangle.
        p2: The third vertex of the triangle.

    Returns:
        A tuple containing the barycentric weights (u, v, w).
    """
    v0x = p1[0] - p0[0]
    v0y = p1[1] - p0[1]
    v1x = p2[0] - p0[0]
    v1y = p2[1] - p0[1]

    d00 = v0x * v0x + v0y * v0y
    d01 = v0x * v1x + v0y * v1y
    d11 = v1x * v1x + v1y * v1y

    v2x = px - p0[0]
    v2y = py - p0[1]

    d20 = v2x * v0x + v2y * v0y
    d21 = v2x * v1x + v2y * v1y

    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return -1.0, -1.0, -1.0

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


@nb.njit(cache=True)
def is_back_facing(p0, p1, p2) -> bool:
    """
    Checks if a triangle is back-facing.

    Args:
        p0: The first vertex of the triangle.
        p1: The second vertex of the triangle.
        p2: The third vertex of the triangle.

    Returns:
        True if the triangle is back-facing, False otherwise.
    """
    edge1 = p1 - p0
    edge2 = p2 - p0
    return float(edge1[0]) * float(edge2[1]) - float(edge1[1]) * float(edge2[0]) >= 0


@nb.njit(cache=True, parallel=True)
def draw_triangle(buffer, zbuffer, p0, p1, p2, c0, c1, c2, w0, w1, w2):
    """
    Draws a filled, textured, and depth-tested triangle.

    Args:
        buffer: The color buffer to draw to.
        zbuffer: The depth buffer for depth testing.
        p0, p1, p2: The screen-space vertices of the triangle.
        c0, c1, c2: The colors of the vertices.
        w0, w1, w2: The w-coordinates of the vertices.
    """
    height, width, channels = buffer.shape
    min_x = max(int(min(p0[0], p1[0], p2[0])), 0)
    max_x = min(int(max(p0[0], p1[0], p2[0])), width - 1)
    min_y = max(int(min(p0[1], p1[1], p2[1])), 0)
    max_y = min(int(max(p0[1], p1[1], p2[1])), height - 1)

    for y in nb.prange(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            u, v, w = barycentric_weights(x + 0.5, y + 0.5, p0, p1, p2)
            if u >= 0 and v >= 0 and w >= 0:

                inv_w0 = 1.0 / w0
                inv_w1 = 1.0 / w1
                inv_w2 = 1.0 / w2

                inv_w_interp = u * inv_w0 + v * inv_w1 + w * inv_w2

                if inv_w_interp == 0:
                    continue

                w_interp = 1.0 / inv_w_interp

                if w_interp < zbuffer[y, x]:
                    zbuffer[y, x] = w_interp

                    for ch in range(channels):
                        val = (u * c0[ch] * inv_w0 + v * c1[ch] *
                               inv_w1 + w * c2[ch] * inv_w2) * w_interp
                        if val < 0:
                            val = 0
                        elif val > 255:
                            val = 255
                        buffer[y, x, ch] = val
