"""
Module realize simple fisheye estimation examples.
"""
import pprint

from resources import EXAMPLE_1, EXAMPLE_O

from lunavl.sdk.estimators.base import ImageWithFaceDetection
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateFisheye():
    """
    Example of a fisheye estimation.

    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    fishEstimator = faceEngine.createFisheyeEstimator()
    faceDetection = detector.detectOne(image, detect5Landmarks=False, detect68Landmarks=True)

    #: single estimation
    imageWithFaceDetection = ImageWithFaceDetection(image, faceDetection.boundingBox)
    fisheye = fishEstimator.estimate(imageWithFaceDetection)
    pprint.pprint(fisheye)

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetection2 = detector.detectOne(image2, detect5Landmarks=False, detect68Landmarks=True)
    #: batch estimation
    imageWithFaceDetectionList = [
        ImageWithFaceDetection(image, faceDetection.boundingBox),
        ImageWithFaceDetection(image2, faceDetection2.boundingBox),
    ]
    fisheyeList = fishEstimator.estimateBatch(imageWithFaceDetectionList)
    pprint.pprint(fisheyeList)


if __name__ == "__main__":
    estimateFisheye()
