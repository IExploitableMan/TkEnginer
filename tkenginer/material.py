import numpy as np

from . import math
from .color import *


class Material:  # TODO: add more built-in materials (lighting, texture)
    def vertex(self, attributes: dict, uniforms: dict) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def fragment(self, varyings: dict, uniforms: dict) -> Color:
        raise NotImplementedError

    def process(self, uniforms: dict, **kwargs) -> None:
        vertices = kwargs["vertices"]
        indices = kwargs["indices"]

        attributes = {
            "position": None,
            "color": None
        }

        varyings_list = [dict() for _ in range(len(vertices))]

        positions_clip = np.empty((len(vertices), 4), dtype=np.float32)

        for i, vertex in enumerate(vertices):
            attributes["position"] = vertex
            attributes["color"] = None

            positions_clip[i], varyings_list[i] = self.vertex(
                attributes, uniforms)

        screen_coords, w_coords = math.clip_to_screen(
            positions_clip,
            uniforms["width"],
            uniforms["height"]
        )

        for triangle in indices:
            w0, w1, w2 = w_coords[triangle[0],
                                  0], w_coords[triangle[1], 0], w_coords[triangle[2], 0]
            if w0 <= 0 or w1 <= 0 or w2 <= 0:
                continue

            p0, p1, p2 = screen_coords[triangle[0]
                                       ], screen_coords[triangle[1]], screen_coords[triangle[2]]

            if math.is_back_facing(p0, p1, p2):
                continue

            c0 = self.fragment(varyings_list[triangle[0]], uniforms).to_numpy()
            c1 = self.fragment(varyings_list[triangle[1]], uniforms).to_numpy()
            c2 = self.fragment(varyings_list[triangle[2]], uniforms).to_numpy()

            math.draw_triangle(
                uniforms["buffer"],
                uniforms["zbuffer"],
                p0, p1, p2,
                c0, c1, c2,
                w0, w1, w2
            )


class MeshColorMaterial(Material):
    def __init__(self, color: Color = Colors.WHITE):
        self.color = color

    def vertex(self, attributes: dict, uniforms: dict) -> tuple[np.ndarray, dict]:
        mvp_matrix = uniforms["mvp_matrix"]
        position = attributes["position"]

        position_clip = math.transform_vertex(position, mvp_matrix)

        varyings = {
            "color": self.color
        }
        return position_clip, varyings

    def fragment(self, varyings: dict, uniforms: dict) -> Color:
        return varyings["color"]
        return varyings["color"]
