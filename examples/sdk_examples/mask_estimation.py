"""
Medical mask estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O
from FaceEngine import DetectionFloat, Rect


def estimateMedicalMask():
    """
    Create warp from detection.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)

    medicalMaskEstimator = faceEngine.createMaskEstimator()
    # Estimate from image
    pprint.pprint(medicalMaskEstimator.estimate(image, faceDetection).asDict())

    # Estimate from wrap
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    pprint.pprint(medicalMaskEstimator.estimate(warp.warpedImage).asDict())


if __name__ == "__main__":
    estimateMedicalMask()
