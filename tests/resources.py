import pathlib

from lunavl.sdk.estimators.face_estimators.emotions import Emotion


def getPathToImage(filename: str) -> str:
    return str(pathlib.Path(__file__).parent.joinpath("data", filename))


CLEAN_ONE_FACE = getPathToImage("girl_front_face.jpg")
ONE_FACE = getPathToImage("one_face.jpg")
WARP_ONE_FACE = getPathToImage("warp_one_face.jpg")
SEVERAL_FACES = getPathToImage("several_faces.jpg")
MANY_FACES = getPathToImage("many_faces.jpg")
NO_FACES = getPathToImage("kand.jpg")
SMALL_IMAGE = getPathToImage("small_image.jpg")
WARP_WHITE_MAN = getPathToImage("warp_white_man.jpg")
HUMAN_WARP = getPathToImage("human_body_warp.jpg")

IMAGE_WITH_TWO_FACES = getPathToImage("2_men.jpg")
IMAGE_WITH_TWO_BODY_ONE_FACE = getPathToImage("2_body_one_face.jpg")

ALL_EMOTIONS = [emotion.name.lower() for emotion in Emotion]
EMOTION_FACES = {emotion: getPathToImage(f"{emotion}.jpg") for emotion in ALL_EMOTIONS}
GOST_HEAD_POSE_FACE = getPathToImage("gost_head_pose.jpg")
TURNED_HEAD_POSE_FACE = getPathToImage("turned_head_pose.jpg")
FRONTAL_HEAD_POSE_FACE = getPathToImage("frontal_head_pose.jpg")

OPEN_EYES = getPathToImage("girl_front_face.jpg")
# https://unsplash.com/photos/RfoISVdKM4U/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8NDZ8fGdpcmxzfGVufDB8fHx8MTY0NzU5NDU3NA&force=true&w=1920
MIXED_EYES = getPathToImage("mixed_eyes.jpg")
CLOSED_EYES = getPathToImage("closed_eyes.jpg")

FACE_WITH_MASK = getPathToImage("face_with_mask.jpg")
OCCLUDED_FACE = getPathToImage("occluded_warp_face.jpg")
MASK_NOT_IN_PLACE = getPathToImage("mask_not_in_place.jpg")
WARP_CLEAN_FACE = getPathToImage("warp_clean_face.jpg")

WARP_FACE_WITH_EYEGLASSES = getPathToImage("warp_face_with_eyeglasses.jpg")
WARP_FACE_WITH_SUNGLASSES = getPathToImage("warp_face_with_sunglasses.jpg")

ROTATED0 = getPathToImage("rotated0.png")
ROTATED90 = getPathToImage("rotated90.png")
ROTATED180 = getPathToImage("rotated180.png")
ROTATED270 = getPathToImage("rotated270.png")
PALETTE_MODE = getPathToImage("palette_mode.png")

BAD_IMAGE = getPathToImage("bad_image.jpg")
BAD_THRESHOLD_WARP = getPathToImage("thumb.jpg")

SPOOF = getPathToImage("spoof.jpg")
UNKNOWN_LIVENESS = getPathToImage("human_body_warp.jpg")
LIVENESS_FACE = getPathToImage("one_face.jpg")
# https://unsplash.com/photos/fVn26rPmO7o
FULL_FACE_WITH_MASK = getPathToImage("medical_mask_face.jpg")
# https://unsplash.com/photos/v6kcLmYIxjo
FULL_OCCLUDED_FACE = getPathToImage("occluded_face.jpg")
# https://unsplash.com/photos/rDEOVtE7vOs
FULL_FACE_NO_MASK = getPathToImage("no_mask_face.jpg")
# https://unsplash.com/photos/IhuHLIxS_Tk
LARGE_IMAGE = getPathToImage("large_image.jpg")

# https://commons.wikimedia.org/wiki/File:Red-eye_effect.png
RED_EYES = getPathToImage("red_eyes.png")

# https://www.pexels.com/ru-ru/photo/11462516/
BLACK_AND_WHITE = getPathToImage("b&w.jpg")

# https://www.pexels.com/ru-ru/photo/11495661/
BASEBALL_CAP = getPathToImage("baseball_cap.jpg")
# https://free-images.com/display/russian_boy_in_cap.html
PEAKED_CAP = getPathToImage("peaked_cap.jpg")
# https://www.pexels.com/ru-ru/photo/10513680/
SHAWL = getPathToImage("shawl.jpg")
# https://unsplash.com/photos/8vT-DgHZDG4
BEANIE = getPathToImage("beanie.jpg")
# https://www.pexels.com/photo/women-standing-back-to-back-7364483/
USHANKA = getPathToImage("ushanka.jpg")
# https://free-images.com/display/cyclist_sports_cycling_man_0.html
HELMET = getPathToImage("helmet.jpg")
# https://unsplash.com/photos/lxEqRDB9Ng8
HOOD = getPathToImage("hood.jpg")
# https://unsplash.com/photos/rr4bawLxOjc
HAT = getPathToImage("hat.jpg")
# https://www.freeimages.com/photo/stupid-1431759
FISHEYE = getPathToImage("fisheye.jpg")
# https://www.freeimages.com/photo/boy-in-swimming-pool-3-1394664
SQUINTING = getPathToImage("squinting.jpg")
# https://free-images.com/display/pouting_boy_in_shamar.html
FROWNING = getPathToImage("frowning.jpg")
# https://www.freeimages.com/ru/download/young-woman-using-computer-elevated-view-portrait-2353499
RAISED = getPathToImage("raised.jpg")
# https://unsplash.com/photos/rwCjbjaK-Zs
BACKPACK = getPathToImage("backpack.jpg")
# https://unsplash.com/photos/u73veL8JTbk
T_SHORT = getPathToImage("t-short.jpg")
# https://unsplash.com/photos/WhnbNX5yeJo
LONG_SLEEVE = getPathToImage("long_sleeve.jpg")
ANGER = getPathToImage("anger.jpg")
# https://unsplash.com/photos/_3ykLzLhOhQ
PINK = getPathToImage("pink.jpg")
# https://unsplash.com/photos/nimElTcTNyY
YELLOW = getPathToImage("yellow.jpg")
# https://unsplash.com/photos/zeh5pg_7H1g
BEIGE = getPathToImage("beige.jpg")
# https://unsplash.com/photos/2rIs8OH5ng0
BLACK = getPathToImage("black.jpg")
# https://unsplash.com/photos/EgZW3jyDlCo
KHAKI = getPathToImage("khaki.jpg")
# https://unsplash.com/photos/KjDAe0dezaY
COLORFUL = getPathToImage("colorful.jpg")
# https://unsplash.com/photos/xgNSlx7DjYM
RED = getPathToImage("red.jpg")
# https://unsplash.com/photos/LL0Rdbcd8X4
MASK_CHIN = getPathToImage("mask_chin.jpg")
# https://free-images.com/display/doctor_medical_medicine_health.html
MASK_MOUTH = getPathToImage("medical_mask_mouth.jpg")
# https://unsplash.com/photos/uCpBPcdK0ew
MASK_FULL = getPathToImage("balaclava.jpg")
