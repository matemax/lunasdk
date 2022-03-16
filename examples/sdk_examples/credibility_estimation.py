"""
Credibility estimation example
"""
import asyncio
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_1


def estimateCredibility():
    """
    Estimate credibility of a person
    """
    image = VLImage.load(filename=EXAMPLE_1)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)

    credibilityEstimator = faceEngine.createCredibilityEstimator()
    pprint.pprint(credibilityEstimator.estimate(warp.warpedImage).asDict())


async def asyncEstimateCredibility():
    """
    Async estimate credibility of a person
    """
    image = VLImage.load(filename=EXAMPLE_1)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)

    credibilityEstimator = faceEngine.createCredibilityEstimator()
    estimation = await credibilityEstimator.estimate(warp.warpedImage, asyncEstimate=True)
    pprint.pprint(estimation.asDict())

    task1 = credibilityEstimator.estimate(warp.warpedImage, asyncEstimate=True)
    task2 = credibilityEstimator.estimate(warp.warpedImage, asyncEstimate=True)
    for task in (task1, task2):
        pprint.pprint(task.get().asDict())


if __name__ == "__main__":
    estimateCredibility()
    asyncio.run(asyncEstimateCredibility())
