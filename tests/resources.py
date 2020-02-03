import pathlib
from lunavl.sdk.estimators.face_estimators.emotions import Emotion


def getPathToImage(filename: str) -> str:
    return str(pathlib.Path(__file__).parent.joinpath("data", filename))


CLEAN_ONE_FACE = getPathToImage("girl_front_face.jpg")
ONE_FACE = getPathToImage("one_face.jpg")
SEVERAL_FACES = getPathToImage("several_faces.jpg")
MANY_FACES = getPathToImage("many_faces.jpg")
NO_FACES = getPathToImage("kand.jpg")
SMALL_IMAGE = getPathToImage("small_image.jpg")

ALL_EMOTIONS = [emotion.name.lower() for emotion in Emotion]
EMOTION_FACES = {emotion: getPathToImage(f"{emotion}.jpg") for emotion in ALL_EMOTIONS}
