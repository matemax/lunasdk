"""
An approximate garbage score estimation example
"""
import pprint

from lunavl.sdk.estimators.base import ImageWithFaceDetection
from resources import EXAMPLE_O, EXAMPLE_1
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateAGS():
    """
    Estimate face detection ags.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)

    agsEstimator = faceEngine.createAGSEstimator()

    imageWithFaceDetection = ImageWithFaceDetection(image, faceDetection.boundingBox)
    pprint.pprint(agsEstimator.estimate(imageWithFaceDetection=imageWithFaceDetection))
    pprint.pprint(agsEstimator.estimate(faceDetection))

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetection2 = detector.detectOne(image2)

    imageWithFaceDetectionList = [
        ImageWithFaceDetection(image, faceDetection.boundingBox),
        ImageWithFaceDetection(image2, faceDetection2.boundingBox),
    ]
    pprint.pprint(agsEstimator.estimateBatch(imageWithFaceDetectionList))

    pprint.pprint(agsEstimator.estimateBatch(detections=[faceDetection, faceDetection2]))


if __name__ == "__main__":
    estimateAGS()
