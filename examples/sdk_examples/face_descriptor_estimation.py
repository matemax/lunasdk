"""
Warp quality estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.facedetector import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateDescriptor():
    """
    Create warp from detection.
    """
    image = VLImage.load(filename='C:/temp/test.jpg')
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createWarper()
    warp = warper.warp(faceDetection)

    extractor = faceEngine.createFaceDescriptorEstimator()

    pprint.pprint(extractor.estimate(warp.warpedImage))
    pprint.pprint(extractor.estimateWarpsBatch([warp.warpedImage, warp.warpedImage]))


if __name__ == "__main__":
    estimateDescriptor()
