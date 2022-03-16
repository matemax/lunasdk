"""
Face descriptor estimate example
"""
import asyncio
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O


def estimateDescriptor():
    """
    Estimate face descriptor.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)

    extractor = faceEngine.createFaceDescriptorEstimator()

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
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)

    extractor = faceEngine.createFaceDescriptorEstimator()

    pprint.pprint(await extractor.estimateDescriptorsBatch([warp.warpedImage, warp.warpedImage], asyncEstimate=True))
    # run tasks and get results
    task1 = extractor.estimate(warp.warpedImage, asyncEstimate=True)
    task2 = extractor.estimate(warp.warpedImage, asyncEstimate=True)
    for task in (task1, task2):
        pprint.pprint(task.get().asDict())


if __name__ == "__main__":
    estimateDescriptor()
    asyncio.run((asyncEstimateDescriptor()))
