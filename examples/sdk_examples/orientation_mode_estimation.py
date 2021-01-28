"""
Module realize simple examples following features:
    * orientation mode estimation
"""
import pprint

from resources import EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.image import VLImage


def estimateOrientationMode():
    """
    Example of a orientation mode estimation.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    orientationModeEstimator = faceEngine.createOrientationModeEstimator()
    #: estimate
    pprint.pprint(orientationModeEstimator.estimate(image))


if __name__ == "__main__":
    estimateOrientationMode()
