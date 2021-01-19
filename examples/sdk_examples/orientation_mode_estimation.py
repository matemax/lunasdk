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
    orientationMode = orientationModeEstimator.estimate(image.coreImage)
    pprint.pprint(orientationMode.asDict())
    pprint.pprint(orientationMode.getOrientationType())


if __name__ == "__main__":
    estimateOrientationMode()
