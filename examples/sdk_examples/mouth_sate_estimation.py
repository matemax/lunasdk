"""
An mouth state estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import FACE_ENGINE
from lunavl.sdk.faceengine.facedetector import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateMouthState():
    """
    Estimate emotion from a warped image.
    """
    image = VLImage.load(filename='C:/temp/test.jpg')
    detector = FACE_ENGINE.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = FACE_ENGINE.createWarper()
    warp = warper.warp(faceDetection)

    emotionEstimator = FACE_ENGINE.createMouthEstimator()

    pprint.pprint(emotionEstimator.estimate(warp.warpedImage).asDict())


if __name__ == "__main__":
    estimateMouthState()
