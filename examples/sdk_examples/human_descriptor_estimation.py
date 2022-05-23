"""
Human descriptor estimate example
"""
import asyncio
import pprint

from resources import EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.image import VLImage


def estimateDescriptor():
    """
    Estimate human descriptor.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()
    humanDetection = detector.detectOne(image)
    warper = faceEngine.createHumanWarper()
    warp = warper.warp(humanDetection)

    extractor = faceEngine.createHumanDescriptorEstimator()

    pprint.pprint(extractor.estimate(warp.warpedImage))
    pprint.pprint(extractor.estimateDescriptorsBatch([warp.warpedImage, warp.warpedImage]))
    batch, aggregateDescriptor = extractor.estimateDescriptorsBatch(
        [warp.warpedImage, warp.warpedImage], aggregate=True
    )
    pprint.pprint(batch)
    pprint.pprint(aggregateDescriptor)


async def asyncEstimateDescriptor():
    """
    Async estimate face descriptor.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()
    humanDetection = detector.detectOne(image)
    warper = faceEngine.createHumanWarper()
    warp = warper.warp(humanDetection)

    extractor = faceEngine.createHumanDescriptorEstimator()

    pprint.pprint(await extractor.estimateDescriptorsBatch([warp.warpedImage, warp.warpedImage], asyncEstimate=True))
    # run tasks and get results
    task1 = extractor.estimate(warp.warpedImage, asyncEstimate=True)
    task2 = extractor.estimate(warp.warpedImage, asyncEstimate=True)
    for task in (task1, task2):
        pprint.pprint(task.get().asDict())


if __name__ == "__main__":
    estimateDescriptor()
    asyncio.run(asyncEstimateDescriptor())
