"""
Glasses estimation example
"""
import asyncio
import pprint

from resources import EXAMPLE_3

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateGlasses():
    """
    Create warp to detection.
    """
    image = VLImage.load(filename=EXAMPLE_3)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)

    glassesEstimator = faceEngine.createGlassesEstimator()
    pprint.pprint(glassesEstimator.estimate(warp.warpedImage).asDict())


async def estimateGlassesAsync():
    """
    Example of an async glasses estimation.

    """
    image = VLImage.load(filename=EXAMPLE_3)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    # async estimation
    glassesEstimator = faceEngine.createGlassesEstimator()
    pprint.pprint((await glassesEstimator.estimate(warp.warpedImage, asyncEstimate=True)).asDict())
    # run tasks and get results
    task1 = glassesEstimator.estimate(warp.warpedImage, asyncEstimate=True)
    task2 = glassesEstimator.estimate(warp.warpedImage, asyncEstimate=True)
    for task in (task1, task2):
        pprint.pprint(task.get().asDict())


if __name__ == "__main__":
    estimateGlasses()
    asyncio.run(estimateGlassesAsync())
