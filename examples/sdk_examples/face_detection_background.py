"""
Module realize simple face detection background estimation examples.
"""
import pprint

from resources import EXAMPLE_4

from lunavl.sdk.estimators.base import ImageWithFaceDetection
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateBackground():
    """
    Example of a face detection background estimation.

    """
    image = VLImage.load(filename=EXAMPLE_4)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    backgroundEstimator = faceEngine.createFaceDetectionBackgroundEstimator()
    faceDetection = detector.detectOne(image, detect5Landmarks=False, detect68Landmarks=True)

    #: single estimation
    imageWithFaceDetection = ImageWithFaceDetection(image, faceDetection.boundingBox)
    background = backgroundEstimator.estimate(imageWithFaceDetection)
    pprint.pprint(background)

    image2 = VLImage.load(filename=EXAMPLE_4)
    faceDetection2 = detector.detectOne(image2, detect5Landmarks=False, detect68Landmarks=True)
    #: batch estimation
    imageWithFaceDetectionList = [
        ImageWithFaceDetection(image, faceDetection.boundingBox),
        ImageWithFaceDetection(image2, faceDetection2.boundingBox),
    ]
    backgrounds = backgroundEstimator.estimateBatch(imageWithFaceDetectionList)
    pprint.pprint(backgrounds)


if __name__ == "__main__":
    estimateBackground()
