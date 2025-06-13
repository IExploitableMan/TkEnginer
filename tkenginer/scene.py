from .gameobject import *


class Scene:
    def __init__(self, gameobjects: list[GameObject] = None) -> None:
        if gameobjects is None:
            gameobjects = list()
        self.gameobjects = gameobjects

    def add(self, gameobject: GameObject) -> None:
        self.gameobjects.append(gameobject)
