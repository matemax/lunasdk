"""
An image color type estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O


def estimateImageColorType():
    """
    Estimate image color type from a warped image.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)

    imageColorTypeEstimator = faceEngine.createImageColorTypeEstimator()

    pprint.pprint(imageColorTypeEstimator.estimate(warp.warpedImage).asDict())


if __name__ == "__main__":
    estimateImageColorType()
