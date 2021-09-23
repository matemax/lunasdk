"""
Medical mask estimation example
"""
import asyncio
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O, EXAMPLE_1


def estimateMedicalMask():
    """
    Medical mask estimation example
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


async def asyncEstimateMedicalMask():
    """
    Async medical mask estimation example
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
    mask = await medicalMaskEstimator.estimate(warp.warpedImage, asyncEstimate=True)
    pprint.pprint(mask.asDict())

    warp2 = warper.warp(detector.detectOne(VLImage.load(filename=EXAMPLE_1)))
    task1 = medicalMaskEstimator.estimate(warp.warpedImage, asyncEstimate=True)
    task2 = medicalMaskEstimator.estimate(warp2.warpedImage, asyncEstimate=True)

    for task in (task1, task2):
        pprint.pprint(task.get())


if __name__ == "__main__":
    estimateMedicalMask()
    asyncio.run(asyncEstimateMedicalMask())
