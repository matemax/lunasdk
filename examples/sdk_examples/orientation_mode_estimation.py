"""
Module realize simple examples following features:
    * orientation mode estimation
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O


def estimateOrientationMode():
    """
    Example of a orientation mode estimation.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    orientationModeEstimator = faceEngine.createOrientationModeEstimator()
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    #: estimate
    orientationMode = orientationModeEstimator.estimate(warp)
    pprint.pprint(orientationMode.asDict())


if __name__ == "__main__":
    estimateOrientationMode()
