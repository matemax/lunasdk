"""
Warps visualization example.
"""
import pprint

import cv2  # pylint: disable=E0611,E0401

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.facedetector import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def createWarp():
    """
    Create warp from detection.

    """
    image = VLImage.load(
        url='https://cdn1.savepice.ru/uploads/2019/4/15/194734af15c4fcd06dec6db86bbeb7cd-full.jpg')
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createWarper()
    warp = warper.warp(faceDetection)
    pprint.pprint(warp.warpedImage.rect)
    cv2.imshow("Wapred image", warp.warpedImage.asNPArray())
    cv2.imshow("Original image", image.asNPArray())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    createWarp()
