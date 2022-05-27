"""
Human descriptor estimate example
"""
import asyncio
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O, EXAMPLE_SEVERAL_FACES


def estimateAttributes():
    """
    Estimate human descriptor.
    """
    image = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()
    humanDetection = detector.detect([image])
    warper = faceEngine.createHumanWarper()
    warp1 = warper.warp(humanDetection[0][0])
    warp2 = warper.warp(humanDetection[0][1])

    estimator = faceEngine.createBodyAttributesEstimator()

    pprint.pprint(estimator.estimate(warp1.warpedImage).asDict())
    pprint.pprint(estimator.estimateBatch([warp1.warpedImage, warp2.warpedImage]))
    aggregated = estimator.aggregate(estimator.estimateBatch([warp1.warpedImage, warp2.warpedImage]))
    pprint.pprint(aggregated)


async def asyncEstimateAttribures():
    """
    Async estimate face descriptor.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()
    humanDetection = detector.detectOne(image)
    warper = faceEngine.createHumanWarper()
    warp = warper.warp(humanDetection)

    estimator = faceEngine.createBodyAttributesEstimator()

    pprint.pprint(await estimator.estimateBatch([warp.warpedImage, warp.warpedImage], asyncEstimate=True))
    # run tasks and get results
    task1 = estimator.estimate(warp.warpedImage, asyncEstimate=True)
    task2 = estimator.estimate(warp.warpedImage, asyncEstimate=True)
    for task in (task1, task2):
        pprint.pprint(task.get().asDict())


if __name__ == "__main__":
    estimateAttributes()
    asyncio.run(asyncEstimateAttribures())
