"""
An approximate garbage score estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import FACE_ENGINE
from lunavl.sdk.faceengine.facedetector import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateAGS():
    """
    Estimate face detection ags.
    """
    image = VLImage.load(filename='C:/temp/test.jpg')
    detector = FACE_ENGINE.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)

    agsEstimator = FACE_ENGINE.createAGSEstimator()

    pprint.pprint(agsEstimator.estimate(image=image, boundingBox=faceDetection.boundingBox))
    pprint.pprint(agsEstimator.estimate(faceDetection))


if __name__ == "__main__":
    estimateAGS()
