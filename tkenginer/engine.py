import tkinter as tk
import numpy as np
import time
import abc

from .scene import *
from . import math


class Engine(abc.ABC):
    def __init__(
        self,
        title: str = "TkEnginer",
        width: int = 1600,
        height: int = 900,
        fps: int = 60,
        subdivision_steps: int = 3,
        fov: float = 90,
        near: float = 0.01,
        far: float = 100,
        clear_color: str = "black",
        scene: Scene = None
    ) -> None:

        self.window = tk.Tk()
        self.window.title(title)
        self.width = width
        self.height = height
        self.window.geometry(f"{width}x{height}")
        self.frame_time = 1000 / fps
        self.subdivision_steps = subdivision_steps
        self.fov = fov
        self.near = near
        self.far = far

        self.projection_matrix = math.get_projection_matrix(
            self.fov, 
            self.width, 
            self.height, 
            self.near, 
            self.far
        )

        self.canvas = tk.Canvas(
            self.window, bg=clear_color, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.yaw = np.pi
        self.pitch = .0
        self.position = np.array([0, 0, 0], dtype=np.float32)

        self.scene = scene if scene is not None else Scene()

        self.pressed_keys: set[str] = set()
        self.mouse: list[int] = None

        self.window.bind("<KeyPress>", self.key_pressed)
        self.window.bind("<KeyRelease>", self.key_released)
        self.window.bind("<ButtonPress>", self.button_pressed)
        self.window.bind("<ButtonRelease>", self.button_released)
        self.window.bind("<Motion>", self.mouse_moved)
        self.window.bind("<Configure>", self.window_resized)

    @abc.abstractmethod
    def on_update(self) -> None:
        pass

    def is_key_pressed(self, key: str) -> bool:
        return key in self.pressed_keys

    def get_mouse_position(self) -> list[int]:
        return self.mouse if self.mouse is not None else [0, 0]

    def run(self) -> None:
        self.update()
        self.window.mainloop()

    def key_pressed(self, event: tk.Event) -> None:
        self.pressed_keys.add(event.keysym)

    def key_released(self, event: tk.Event) -> None:
        self.pressed_keys.discard(event.keysym)

    def button_pressed(self, event: tk.Event) -> None:
        self.pressed_keys.add(f"mouse_{event.num}")

    def button_released(self, event: tk.Event) -> None:
        self.pressed_keys.discard(f"mouse_{event.num}")

    def mouse_moved(self, event: tk.Event) -> None:
        self.mouse = [event.x, event.y]

    def window_resized(self, event: tk.Event) -> None:
        self.width, self.height = event.width, event.height
        self.projection_matrix = math.get_projection_matrix(
            self.fov,
            self.width, 
            self.height, 
            self.near, 
            self.far
        )

    def update(self) -> None: # TODO: depth test, shaders, color blending
        t = time.time()
        self.on_update()

        self.canvas.delete(tk.ALL)

        view_matrix = math.get_view_matrix(self.position, self.yaw, self.pitch)

        for gameobject in self.scene.gameobjects:
            vertices, indices = gameobject.mesh.get_data()
            mvp_matrix = self.projection_matrix @ view_matrix @ gameobject.transform.get_matrix()

            vertices_clip = math.transform_vertices(vertices, mvp_matrix)
            screen_coords, w_coords = math.clip_to_screen(
                vertices_clip,
                self.width,
                self.height
            )

            for triangle in indices:
                w0, w1, w2 = w_coords[triangle[0], 0], w_coords[triangle[1], 0], w_coords[triangle[2], 0]
                if w0 <= 0 or w1 <= 0 or w2 <= 0:
                    continue

                p0, p1, p2 = screen_coords[triangle[0]], screen_coords[triangle[1]], screen_coords[triangle[2]]

                edge1 = p1 - p0
                edge2 = p2 - p0
                if edge1[0] * edge2[1] - edge1[1] * edge2[0] >= 0:
                    continue
                
                if self.subdivision_steps > 0:
                    math.draw_subdivided_triangle(
                        p0,
                        p1,
                        p2,
                        (255, 0, 0),
                        (0, 255, 0),
                        (0, 0, 255),
                        self.subdivision_steps,
                        self.canvas
                    )
                else:
                    self.canvas.create_polygon(
                        [p0[0], p0[1], p1[0], p1[1], p2[0], p2[1]],
                        outline="white",
                        fill=""
                    )
        self.window.after(
            max(1, int(self.frame_time - 1000 * (time.time() - t))),
            self.update
        )
