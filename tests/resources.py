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
WARP_WHITE_MAN = getPathToImage("warp_white_man.jpg")
HUMAN_WARP = getPathToImage("human_body_warp.jpg")

ALL_EMOTIONS = [emotion.name.lower() for emotion in Emotion]
EMOTION_FACES = {emotion: getPathToImage(f"{emotion}.jpg") for emotion in ALL_EMOTIONS}
GOST_HEAD_POSE_FACE = getPathToImage("gost_head_pose.jpg")
TURNED_HEAD_POSE_FACE = getPathToImage("turned_head_pose.jpg")
FRONTAL_HEAD_POSE_FACE = getPathToImage("frontal_head_pose.jpg")

OPEN_EYES = getPathToImage("girl_front_face.jpg")
MIXED_EYES = getPathToImage("one_face.jpg")
CLOSED_EYES = getPathToImage("closed_eyes.jpg")

FACE_WITH_MASK = getPathToImage("mask.jpg")