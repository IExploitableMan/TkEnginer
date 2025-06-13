from .transform import *
from .mesh import *


class GameObject:  # TODO: children/nodes, not gameobjs etc.
    def __init__(self, mesh: Mesh, transform=None) -> None:
        if transform is None:
            self.transform = Transform()
        else:
            self.transform = transform
        self.mesh = mesh
