"""
Tests for the node module.
"""

from tkenginer.node import *
from tkenginer.transform import *
from tkenginer.mesh import *
from tkenginer.material import *

def test_node_init_defaults():
    """
    Tests the default initialization of a Node.
    """
    node = Node()
    assert node.mesh is None
    assert isinstance(node.material, MeshColorMaterial)
    assert isinstance(node.transform, Transform)
    assert node.children == []

def test_node_init_with_values():
    """
    Tests the initialization of a Node with specified values.
    """
    mesh = Mesh([], [])
    material = Material()
    transform = Transform()
    children = [Node(), Node()]
    node = Node(mesh=mesh, material=material, transform=transform, children=children)
    assert node.mesh is mesh
    assert node.material is material
    assert node.transform is transform
    assert node.children is children

def test_node_update():
    """
    Tests the update method of a Node.
    """
    node = Node()
    node.update(0.1)

def test_node_traverse_single():
    """
    Tests traversing a single node.
    """
    transform = Transform(position=[1, 0, 0])
    node = Node(transform=transform)
    
    nodes = list(node.traverse())
    
    assert len(nodes) == 1
    assert nodes[0][0] is node
    assert nodes[0][1] == transform

def test_node_traverse_with_children():
    """
    Tests traversing a node with children.
    """
    child_transform = Transform(position=[0, 1, 0])
    child = Node(transform=child_transform)
    
    parent_transform = Transform(position=[1, 0, 0])
    parent = Node(transform=parent_transform, children=[child])
    
    nodes = list(parent.traverse())
    
    assert len(nodes) == 2
    assert nodes[0][0] is parent
    assert nodes[0][1] == parent_transform
    
    assert nodes[1][0] is child
    
    expected_global_transform = parent_transform @ child_transform
    assert nodes[1][1] == expected_global_transform
