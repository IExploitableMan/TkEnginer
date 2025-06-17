import numpy as np

from .node import Node
from .transform import Transform
from .mesh import Mesh

gravity = np.array([0, -9.81, 0], dtype=np.float32)


class RigidbodyNode(Node):
    def __init__(
        self,
        mesh: Mesh = None,
        transform: Transform = None,
        children: list = None,
        mass: float = 1.0,
        use_gravity: bool = True,
    ) -> None:
        super().__init__(mesh, transform, children)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.mass = mass
        self.use_gravity = use_gravity

    def apply_force(self, force: np.ndarray):
        self.acceleration += force / self.mass

    def update(self, delta: float) -> None:
        if self.use_gravity:
            self.acceleration += gravity
        self.velocity += self.acceleration * delta
        self.transform.position += self.velocity * delta
        self.acceleration[:] = 0
