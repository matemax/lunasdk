"""
Module realize simple examples following features:
    * one face detection
    * batch images face detection
    * detect landmarks68 and landmarks5
"""
import pprint

from lunavl.sdk.faceengine.engine import FACE_ENGINE
from lunavl.sdk.faceengine.facedetector import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def detectOneFace():
    """
    Detect one face on an image.
    """
    detector = FACE_ENGINE.createFaceDetector(DetectorType.FACE_DET_V1)

    imageWithOneFace = VLImage.load(
        url='https://cdn1.savepice.ru/uploads/2019/4/15/194734af15c4fcd06dec6db86bbeb7cd-full.jpg')
    pprint.pprint(detector.detectOne(imageWithOneFace, detect5Landmarks=False, detect68Landmarks=False).asDict())

    imageWithSeveralFaces = VLImage.load(
        url='https://cdn1.savepice.ru/uploads/2019/4/15/aa970957128d9892f297cdfa5b3fda88-full.jpg')
    pprint.pprint(detector.detectOne(imageWithSeveralFaces, detect5Landmarks=False, detect68Landmarks=False).asDict())

    imageWithoutFace = VLImage.load(
        url='https://cdn1.savepice.ru/uploads/2019/4/15/3e3593dc2fd0671c7051b18c99974192-full.jpg')
    pprint.pprint(detector.detectOne(imageWithoutFace, detect5Landmarks=False, detect68Landmarks=False) is None)


if __name__ == "__main__":
    detectOneFace()
