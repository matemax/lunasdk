import pprint

from lunavl.sdk.faceengine.engine import FACE_ENGINE
from lunavl.sdk.faceengine.facedetector import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def createWarp():
    """
    Create warp from detection.

    """
    image = VLImage.load(
        url='https://cdn1.savepice.ru/uploads/2019/4/15/194734af15c4fcd06dec6db86bbeb7cd-full.jpg')
    detector = FACE_ENGINE.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = FACE_ENGINE.createWarper()
    warp = warper.warp(faceDetection)

    qualityEstimator = FACE_ENGINE.createImageQualityEstimator()

    pprint.pprint(qualityEstimator.estimate(warp.warpedImage))


if __name__ == "__main__":
    createWarp()
