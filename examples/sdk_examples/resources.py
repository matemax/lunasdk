import os
from pathlib import Path


def getPathToImage(filename: str) -> str:
    exampleResourcesDirDir = os.path.dirname(os.path.abspath(__file__))
    pathToFile = Path(exampleResourcesDirDir).joinpath("resources", filename)
    if not pathToFile.exists():
        raise FileNotFoundError(pathToFile)
    return str(pathToFile)


EXAMPLE_O = getPathToImage("example_0.jpg")
EXAMPLE_1 = getPathToImage("example_1.jpg")
EXAMPLE_2 = getPathToImage("example_2.jpg")
EXAMPLE_3 = getPathToImage("example_3.jpg")
EXAMPLE_WITHOUT_FACES = getPathToImage("without_faces.jpg")
EXAMPLE_SEVERAL_FACES = getPathToImage("several_faces.jpg")
