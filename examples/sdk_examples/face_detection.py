"""
Module realize simple examples following features:
    * one face detection
    * batch images face detection
    * detect landmarks68 and landmarks5
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_WITHOUT_FACES, EXAMPLE_SEVERAL_FACES, EXAMPLE_O


def detectOneFace():
    """
    Detect one face on an image.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)

    imageWithOneFace = VLImage.load(
        filename=EXAMPLE_O
    )
    pprint.pprint(detector.detectOne(imageWithOneFace, detect5Landmarks=False, detect68Landmarks=False).asDict())

    imageWithSeveralFaces = VLImage.load(
        filename=EXAMPLE_SEVERAL_FACES
    )
    pprint.pprint(detector.detectOne(imageWithSeveralFaces, detect5Landmarks=False, detect68Landmarks=False).asDict())

    severalFaces = detector.detect([imageWithSeveralFaces], detect5Landmarks=False, detect68Landmarks=False)
    pprint.pprint([face.asDict() for face in severalFaces[0]])
    imageWithoutFace = VLImage.load(
        filename=EXAMPLE_WITHOUT_FACES
    )
    pprint.pprint(detector.detectOne(imageWithoutFace, detect5Landmarks=False, detect68Landmarks=False) is None)

if __name__ == "__main__":
    detectOneFace()
