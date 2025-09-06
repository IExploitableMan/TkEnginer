import tkinter as tk
import numpy as np
import time

from PIL import Image, ImageTk
from .node import *
from . import math
from .color import *


class Engine:
    def __init__(
        self,
        title: str = "TkEnginer",
        width: int = 1600,
        height: int = 900,
        fps: int = 60,
        fov: float = 90,
        near: float = 0.01,
        far: float = 100,
        clear_color: Color = Colors.BLACK,
        scene: Node = None
    ) -> None:

        self.window = tk.Tk()
        self.window.title(title)
        self.window.geometry(f"{width}x{height}")
        self.frame_time = 1000 / fps
        self.fov = fov
        self.near = near
        self.far = far
        self.clear_color = clear_color

        self.canvas = tk.Canvas(
            self.window,
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.init(width, height)
        self.last_time = time.time()

        self.yaw = np.pi
        self.pitch = .0
        self.position = np.array([0, 0, 0], dtype=np.float32)

        self.scene = scene if scene is not None else Node()

        self.pressed_keys: set[str] = set()
        self.mouse: list[int] = None

        self.window.bind("<KeyPress>", self.key_pressed)
        self.window.bind("<KeyRelease>", self.key_released)
        self.window.bind("<ButtonPress>", self.button_pressed)
        self.window.bind("<ButtonRelease>", self.button_released)
        self.window.bind("<Motion>", self.mouse_moved)
        self.window.bind("<Configure>", self.window_resized)

    def init(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.projection_matrix = math.get_projection_matrix(
            self.fov,
            self.width,
            self.height,
            self.near,
            self.far
        )
        self.buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.zbuffer = np.full((self.height, self.width),
                               np.inf, dtype=np.float32)
        self.image = Image.fromarray(self.buffer, "RGBA")
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def update(self, delta: float) -> None:
        pass

    def is_key_pressed(self, key: str) -> bool:
        return key in self.pressed_keys

    def get_mouse_position(self) -> list[int]:
        return self.mouse if self.mouse is not None else [0, 0]

    def run(self) -> None:
        self.loop()
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
        self.init(event.width, event.height)

    def loop(self) -> None:  # TODO: lighting
        now = time.time()
        delta = now - self.last_time

        self.image.paste("black", (0, 0, self.width, self.height))
        self.buffer[:, :, :] = self.clear_color.to_tuple()
        self.zbuffer[:, :] = np.inf

        view_matrix = math.get_view_matrix(self.position, self.yaw, self.pitch)

        for node, global_transform in self.scene.traverse():
            node.update(delta)
            if node.mesh is None:
                continue

            mvp_matrix = self.projection_matrix @ view_matrix @ global_transform.get_matrix()

            uniforms = {
                "mvp_matrix": mvp_matrix,
                "width": self.width,
                "height": self.height,
                "buffer": self.buffer,
                "zbuffer": self.zbuffer
            }

            vertices, indices = node.mesh.get_data()

            node.material.process(
                uniforms,
                vertices=vertices,
                indices=indices,
                colors=None
            )

        self.image = Image.fromarray(self.buffer, "RGBA")
        self.update(delta)
        self.photo.paste(self.image)

        self.last_time = now
        self.window.after(
            max(1, int(self.frame_time - 1000 * (time.time() - now))),
            self.loop
        )
