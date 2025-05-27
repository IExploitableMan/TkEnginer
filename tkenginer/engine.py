import tkinter as tk
import abc
import time
import numpy as np

from .scene import Scene
from .transform import Transform


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
        clear_color: str = "black",
        scene: Scene = None
    ) -> None:

        self.window = tk.Tk()
        self.window.title(title)
        self.width = width
        self.height = height
        self.window.geometry(f"{width}x{height}")
        self.frame_time = 1000 / fps
        self.fov = fov
        self.near = near
        self.far = far
        self.update_projection_matrix()

        self.canvas = tk.Canvas(
            self.window,
            bg=clear_color,
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.yaw, self.pitch = np.pi, 0.0  # TODO: better camera
        self.position = np.array([0, 0, 0], dtype=np.float32)

        if Scene is None:
            self.scene = Scene()
        else:
            self.scene = scene

        self.pressed_keys = set[str]()
        self.mouse = None

        self.window.bind("<KeyPress>", self.key_pressed)
        self.window.bind("<KeyRelease>", self.key_released)
        self.window.bind("<ButtonPress>", self.button_pressed)
        self.window.bind("<ButtonRelease>", self.button_released)
        self.window.bind("<Motion>", self.mouse_moved)

    @abc.abstractmethod
    def on_update(self) -> None:
        ...

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
        if event.keysym in self.pressed_keys:
            self.pressed_keys.remove(event.keysym)

    def button_pressed(self, event: tk.Event) -> None:
        self.pressed_keys.add(f"mouse_{event.num}")

    def button_released(self, event: tk.Event) -> None:
        if f"mouse_{event.num}" in self.pressed_keys:
            self.pressed_keys.remove(f"mouse_{event.num}")

    def mouse_moved(self, event: tk.Event) -> None:
        self.mouse = [event.x, event.y]

    def window_resized(self, event: tk.Event) -> None:
        self.width, self.height = event.width, event.height
        self.update_projection_matrix()

    def update_projection_matrix(self) -> None:
        focal = 1 / np.tan(np.radians(self.fov) / 2)
        self.projection_matrix = np.zeros((4, 4), dtype=np.float32)
        self.projection_matrix[0, 0] = focal / (self.width / self.height)
        self.projection_matrix[1, 1] = focal
        self.projection_matrix[2, 2] = (self.far + self.near) / (self.near - self.far)
        self.projection_matrix[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        self.projection_matrix[3, 2] = -1

    def update(self) -> None:
        t = time.time()
        self.on_update()

        self.canvas.delete(tk.ALL)

        front = np.array([
            np.cos(self.pitch) * np.sin(self.yaw),
            np.sin(self.pitch),
            np.cos(self.pitch) * np.cos(self.yaw)
        ], dtype=np.float32)
        front /= np.linalg.norm(front)
        right = np.cross([0, 1, 0], front)
        right /= np.linalg.norm(right)
        up = np.cross(front, right)

        view_matrix = np.identity(4, dtype=np.float32)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = up
        view_matrix[2, :3] = -front  # OpenGL standard...
        view_matrix[:3, 3] = -view_matrix[:3, :3] @ self.position

        for gameobject in self.scene.gameobjects:
            vertices, indices = gameobject.mesh.get_data()
            mvp_matrix = self.projection_matrix @ view_matrix @ gameobject.transform.get_matrix()

            vertices_homogeneous = np.hstack(
                [vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)])

            vertices_clip = vertices_homogeneous @ mvp_matrix.T

            w_coords = vertices_clip[:, 3:4]

            vertices_ndc = vertices_clip[:, :3] / w_coords

            screen_coords = np.empty((len(vertices_ndc), 2), dtype=np.int32)
            screen_coords[:, 0] = (
                (vertices_ndc[:, 0] + 1) * 0.5 * self.width).astype(np.int32)
            screen_coords[:, 1] = (
                (1 - (vertices_ndc[:, 1] + 1) * 0.5) * self.height).astype(np.int32)

            for triangle in indices:
                w0, w1, w2 = w_coords[triangle[0],
                                      0], w_coords[triangle[1], 0], w_coords[triangle[2], 0]
                if w0 <= 0 or w1 <= 0 or w2 <= 0:
                    continue

                p0, p1, p2 = screen_coords[triangle[0]
                                           ], screen_coords[triangle[1]], screen_coords[triangle[2]]

                points = [
                    p0[0], p0[1],
                    p1[0], p1[1],
                    p2[0], p2[1],
                ]
                # TODO: "fragment" shader
                self.canvas.create_polygon(points, outline="white", fill="")

        self.window.after(
            max(1, int(self.frame_time - 1000 * (time.time() - t))), self.update)
