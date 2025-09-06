import numpy as np

from .node import Node
from .transform import Transform
from .mesh import Mesh

gravity = np.array([0, -9.81, 0], dtype=np.float32)


class OBB:
    def __init__(
        self,
        center: np.ndarray = None,
        axes: np.ndarray = None,
        extents: np.ndarray = None
    ) -> None:
        self.center = center if center is not None else np.zeros(
            3, dtype=np.float32)
        self.axes = axes if axes is not None else np.eye(
            3, dtype=np.float32)
        self.extents = extents if extents is not None else np.ones(
            3, dtype=np.float32)

    def update(self, center, axes, extents) -> None:
        self.center = center
        self.axes = axes
        self.extents = extents

    def collides_with(self, other: 'OBB') -> bool:
        R = self.axes.T @ other.axes
        T = self.axes.T @ (other.center - self.center)

        eps = np.finfo(np.float32).eps * 3
        abs_R = np.abs(R) + eps

        ra = self.extents
        rb = (other.extents[None, :] * abs_R).sum(axis=1)
        if np.any(np.abs(T) > ra + rb):
            return False

        ra = (self.extents[None, :] * abs_R).sum(axis=0)
        rb = other.extents
        proj = np.abs((T @ R))
        if np.any(proj > ra + rb):
            return False

        for i in range(3):
            for j in range(3):
                ra_edge = (
                    self.extents[(i+1) % 3] * abs_R[(i+2) % 3, j] +
                    self.extents[(i+2) % 3] * abs_R[(i+1) % 3, j]
                )
                rb_edge = (
                    other.extents[(j+1) % 3] * abs_R[i, (j+2) % 3] +
                    other.extents[(j+2) % 3] * abs_R[i, (j+1) % 3]
                )
                t = abs(
                    T[(i+2) % 3] * R[(i+1) % 3, j] -
                    T[(i+1) % 3] * R[(i+2) % 3, j]
                )
                if t > ra_edge + rb_edge:
                    return False

        return True

# TODO: switch to ECS
# class RigidbodyNode(Node): 
#     def __init__(
#         self,
#         mesh: Mesh = None,
#         transform: Transform = None,
#         children: list = None,
#         mass: float = 1.0,
#         use_gravity: bool = True,
#         obb: OBB = None
#     ) -> None:
#         super().__init__(mesh, transform, children)
#         self.velocity = np.zeros(3, dtype=np.float32)
#         self.acceleration = np.zeros(3, dtype=np.float32)
#         self.mass = mass
#         self.use_gravity = use_gravity
#         self.obb = obb

#     def apply_force(self, force: np.ndarray) -> None:
#         self.acceleration += force / self.mass

#     def update(self, delta: float) -> None:
#         if self.use_gravity:
#             self.acceleration += gravity
#         self.velocity += self.acceleration * delta
#         self.transform.position += self.velocity * delta
#         self.acceleration[:] = 0
