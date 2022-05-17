"""
A heawear estimation example
"""
import pprint

from resources import EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateHeadwear():
    """
    Estimate headwear from a warped image.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)

    eyebrowEstimator = faceEngine.createHeadwearEstimator()

    pprint.pprint(eyebrowEstimator.estimate(warp.warpedImage).asDict())


if __name__ == "__main__":
    estimateHeadwear()
