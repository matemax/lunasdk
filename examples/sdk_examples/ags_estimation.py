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
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)

    agsEstimator = faceEngine.createAGSEstimator()

    imageWithFaceDetection = ImageWithFaceDetection(image.coreImage, faceDetection.boundingBox.coreEstimation)
    pprint.pprint(agsEstimator.estimate(imageWithFaceDetection=imageWithFaceDetection))
    pprint.pprint(agsEstimator.estimate(faceDetection))

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetection2 = detector.detectOne(image2)

    imageWithFaceDetectionList = [
        ImageWithFaceDetection(image.coreImage, faceDetection.boundingBox.coreEstimation),
        ImageWithFaceDetection(image2.coreImage, faceDetection2.boundingBox.coreEstimation),
    ]
    pprint.pprint(agsEstimator.estimateAgsBatchByImages(imageWithFaceDetectionList))

    pprint.pprint(agsEstimator.estimateAgsBatchByDetections(detections=[faceDetection, faceDetection2]))


if __name__ == "__main__":
    estimateAGS()
