import copy

from .transform import *
from .mesh import *


class Node:
    def __init__(self, mesh: Mesh = None, transform: Transform = None, children: "Node" = None) -> None:
        self.mesh = mesh
        self.transform = transform if transform is not None else Transform()
        self.children = children if children is not None else list()

    def update(self, delta: float) -> None:
        pass

    def traverse(self, parent_transform=Transform()):
        global_transform = parent_transform @ self.transform
        yield self, global_transform
        for child in self.children:
            yield from child.traverse(global_transform)

