from .transform import *
from .mesh import *


class Node:
    def __init__(self, mesh: Mesh = None, transform: Transform = None, children: "Node" = None) -> None:
        self.mesh = mesh
        if transform is None:
            self.transform = Transform()
        else:
            self.transform = transform
        if children is None:
            self.children: list[Node] = list()
        else:
            self.children = children

    def flatten(self, parent_transform: Transform = None) -> list["Node"]:
        if parent_transform is None:
            parent_transform = Transform()
        global_transform = parent_transform @ self.transform
        result = [Node(self.mesh, global_transform, [])]
        for child in self.children:
            result.extend(child.flatten(global_transform))
        return result

