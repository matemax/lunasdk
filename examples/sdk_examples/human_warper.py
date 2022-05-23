"""
Creating human warp example.
"""
import pprint

from resources import EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.image import VLImage


def createWarp():
    """
    Create human body warp from human detection.

    """
    faceEngine = VLFaceEngine()
    image = VLImage.load(filename=EXAMPLE_O)
    detector = faceEngine.createHumanDetector()
    humanDetection = detector.detectOne(image)
    warper = faceEngine.createHumanWarper()
    warp = warper.warp(humanDetection)
    pprint.pprint(warp.warpedImage.rect)


if __name__ == "__main__":
    createWarp()
