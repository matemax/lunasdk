import pathlib


def getPathToImage(filename: str) -> str:
    return str(pathlib.Path(__file__).parent.joinpath('data', filename))


ONE_FACE = getPathToImage('one_face.jpg')
SEVERAL_FACES = getPathToImage('several_faces.jpg')
MANY_FACES = getPathToImage('many_faces.jpg')
NO_FACES = getPathToImage('kand.jpg')
