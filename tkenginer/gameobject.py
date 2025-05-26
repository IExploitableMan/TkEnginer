from .transform import Transform
from .mesh import Mesh


class GameObject:  # TODO: children
    def __init__(self, mesh: Mesh, transform=None) -> None:
        if transform is None:
            self.transform = Transform()
        else:
            self.transform = transform
        self.mesh = mesh
