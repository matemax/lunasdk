"""
Warp quality estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import FACE_ENGINE
from lunavl.sdk.faceengine.facedetector import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateDescriptor():
    """
    Create warp from detection.
    """
    image = VLImage.load(filename='C:/temp/test.jpg')
    detector = FACE_ENGINE.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = FACE_ENGINE.createWarper()
    warp = warper.warp(faceDetection)

    extractor = FACE_ENGINE.createFaceDescriptorEstimator()

    pprint.pprint(extractor.estimate(warp.warpedImage))
    pprint.pprint(extractor.estimateWarpsBatch([warp.warpedImage, warp.warpedImage]))



if __name__ == "__main__":
    estimateDescriptor()
