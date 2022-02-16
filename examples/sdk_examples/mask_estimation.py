"""
Medical mask estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O, EXAMPLE_1


def estimateMedicalMask():
    """
    Create warp from detection.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)

    medicalMaskEstimator = faceEngine.createMaskEstimator()
    # Estimate from detection
    pprint.pprint(medicalMaskEstimator.estimate(faceDetection).asDict())

    # Estimate from wrap
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    pprint.pprint(medicalMaskEstimator.estimate(warp.warpedImage).asDict())

    warp2 = warper.warp(detector.detectOne(VLImage.load(filename=EXAMPLE_1)))

    pprint.pprint(medicalMaskEstimator.estimateBatch([warp, warp2]))


if __name__ == "__main__":
    estimateMedicalMask()
