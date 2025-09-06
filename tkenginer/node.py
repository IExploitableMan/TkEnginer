from .transform import *
from .material import *
from .mesh import *


class Node:
    def __init__(self, mesh: Mesh = None, material: Material = None, transform: Transform = None, children: "Node" = None) -> None:
        self.mesh = mesh
        self.material = material if material is not None else MeshColorMaterial()
        self.transform = transform if transform is not None else Transform()
        self.children = children if children is not None else list()

    def update(self, delta: float) -> None:
        pass

    def traverse(self, parent_transform=Transform()):
        global_transform = parent_transform @ self.transform
        yield self, global_transform
        for child in self.children:
            yield from child.traverse(global_transform)
