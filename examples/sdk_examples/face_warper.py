"""
Creating warp example.
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O


def createWarp():
    """
    Create face warp from detection.

    """
    faceEngine = VLFaceEngine()
    image = VLImage.load(filename=EXAMPLE_O)
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    pprint.pprint(warp.warpedImage.rect)


if __name__ == "__main__":
    createWarp()
