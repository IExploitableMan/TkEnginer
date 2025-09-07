"""
This module provides the Node class, which is the basic building block of a scene graph.
"""

from .transform import *
from .material import *
from .mesh import *


class Node:
    """
    A node in the scene graph.
    """

    def __init__(self, mesh: Mesh = None, material: Material = None, transform: Transform = None, children: "Node" = None) -> None:
        """
        Initializes the node.

        Args:
            mesh: The mesh to be rendered for this node.
            material: The material to use for rendering the mesh.
            transform: The local transform of this node.
            children: A list of child nodes.
        """
        self.mesh = mesh
        self.material = material if material is not None else MeshColorMaterial()
        self.transform = transform if transform is not None else Transform()
        self.children = children if children is not None else list()

    def update(self, delta: float) -> None:
        """
        Called every frame.

        Args:
            delta: The time since the last frame.
        """
        pass

    def traverse(self, parent_transform=Transform()):
        """
        Traverses the scene graph starting from this node.

        Args:
            parent_transform: The transform of the parent node.

        Yields:
            A tuple containing the node and its global transform.
        """
        global_transform = parent_transform @ self.transform
        yield self, global_transform
        for child in self.children:
            yield from child.traverse(global_transform)
