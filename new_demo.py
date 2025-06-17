import tkenginer as tke
import numpy as np
import time

from PIL import ImageDraw


class Demo(tke.Engine):
    def __init__(self) -> None:
        super().__init__("TkEnginer new demo")
        self.speed = 3
        self.sensetivity = 0.2
        self.last_mouse = None

    def update(self, delta: float) -> None:
        draw = ImageDraw.Draw(self.image)
        draw.text(
            (10, 10), 
            f"FPS: {int(1 / delta)}", 
            "white"
        )
        front, right, _ = tke.math.get_camera_vectors(self.yaw, self.pitch)

        mouse = self.get_mouse_position()
        if self.is_key_pressed("mouse_1"):
            self.yaw += (mouse[0] - self.last_mouse[0]) * self.sensetivity * delta
            self.pitch -= (mouse[1] - self.last_mouse[1]) * self.sensetivity * delta
            max_pitch = np.pi / 2 - 0.01
            self.pitch = max(-max_pitch, min(max_pitch, self.pitch))
        self.last_mouse = mouse

        if self.is_key_pressed("w"):
            self.position += front * self.speed * delta
        if self.is_key_pressed("a"):
            self.position -= right * self.speed * delta
        if self.is_key_pressed("s"):
            self.position -= front * self.speed * delta
        if self.is_key_pressed("d"):
            self.position += right * self.speed * delta


demo = Demo()
demo.scene = tke.Node(children=[
    tke.Node(
        mesh=tke.SphereMesh(8),
        transform=tke.Transform(
            position=[2.0, -3.0, -3.0],
            rotation=[np.pi/4, np.pi/2, np.pi/3],
            scale=[0.7, 0.5, 1.2]
        )
    ),
    tke.Node(
        mesh=tke.CubeMesh(
            colors=[
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
                [255, 255, 0, 255],
                [255, 0, 255, 255],
                [0, 255, 255, 255],
                [255, 165, 0, 255],
                [128, 0, 128, 255],
            ]
        ),
        transform=tke.Transform(
            position=[-5.0, 2.0, -3.0],
            rotation=[np.pi/6, np.pi/3, np.pi/4],
            scale=[1.5, 0.5, 1.0]
        )
    ),
    tke.RigidbodyNode(
        mesh=tke.PyramidWithSquareBaseMesh(),
        transform=tke.Transform(
            position=[4.0, 3.0, -3.0],
            rotation=[np.pi/3, np.pi/2, np.pi/5],
            scale=[0.8, 1.0, 0.5]
        ),
        mass=0.1
    ),
    tke.Node(
        mesh=tke.PyramidMesh(),
        transform=tke.Transform(
            position=[-2.0, -1.0, -3.0],
            rotation=[np.pi/4, np.pi/6, np.pi/2],
            scale=[1.0, 1.3, 0.6]
        )
    ),
    tke.Node(
        mesh=tke.ConeMesh(15),
        transform=tke.Transform(
            position=[1.0, 5.0, -3.0],
            rotation=[np.pi/2, np.pi/4, np.pi/6],
            scale=[1.0, 0.8, 1.5]
        )
    ),
    tke.Node(
        mesh=tke.PlaneMesh(),
        transform=tke.Transform(
            position=[0.0, -2.0, -3.0],
            rotation=[0.0, 0.0, 0.0],
            scale=[3.0, 3.0, 3.0]
        )
    ),
    # tke.GameObject(
    #     mesh=tke.OBJLoader(open("ak74.obj", "r")),
    #     transform=tke.Transform(
    #         position=[-1.0, 0.0, 3.0],
    #         rotation=[np.pi/4, np.pi/2, np.pi/3],
    #         scale=[0.1, 0.1, 0.1]
    #     )
    # )
])
demo.run()
