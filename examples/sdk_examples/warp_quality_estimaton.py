"""
Warp quality estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.facedetector import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateWarpQuality():
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

    qualityEstimator = faceEngine.createWarpQualityEstimator()

    pprint.pprint(qualityEstimator.estimate(warp.warpedImage).asDict())


if __name__ == "__main__":
    estimateWarpQuality()
