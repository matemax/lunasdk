"""
Creating warp example.
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def createWarp():
    """
    Create warp from detection.

    """
    faceEngine = VLFaceEngine()
    image = VLImage.load(url="https://cdn1.savepice.ru/uploads/2019/4/15/194734af15c4fcd06dec6db86bbeb7cd-full.jpg")
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createWarper()
    warp = warper.warp(faceDetection)
    pprint.pprint(warp.warpedImage.rect)


if __name__ == "__main__":
    createWarp()
