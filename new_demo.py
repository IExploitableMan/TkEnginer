import tkenginer


class Demo(tkenginer.Engine):
    def __init__(self):
        super().__init__("TkEnginer new demo")

    def on_update(self):

        if self.is_key_pressed("w"):
            print("W key pressed")
        if self.is_key_pressed("mouse_1"):
            print("Left mouse button pressed")


demo = Demo()
demo.scene = tkenginer.Scene([
    tkenginer.GameObject(
        mesh=tkenginer.SphereMesh(24),
        transform=tkenginer.Transform(
            position=[0, 0, -3])
    )
])
demo.run()
