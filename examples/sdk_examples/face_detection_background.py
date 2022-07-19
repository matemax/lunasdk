"""
Module realize simple face detection background estimation examples.
"""
import asyncio
import pprint

from resources import EXAMPLE_4

from lunavl.sdk.estimators.base import ImageWithFaceDetection
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateBackground():
    """
    Example of a face detection background estimation.

    """
    image = VLImage.load(filename=EXAMPLE_4)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    backgroundEstimator = faceEngine.createFaceDetectionBackgroundEstimator()
    faceDetection = detector.detectOne(image)

    #: single estimation
    imageWithFaceDetection = ImageWithFaceDetection(image, faceDetection.boundingBox)
    background = backgroundEstimator.estimate(imageWithFaceDetection)
    pprint.pprint(background)

    image2 = VLImage.load(filename=EXAMPLE_4)
    faceDetection2 = detector.detectOne(image2)
    #: batch estimation
    imageWithFaceDetectionList = [
        ImageWithFaceDetection(image, faceDetection.boundingBox),
        ImageWithFaceDetection(image2, faceDetection2.boundingBox),
    ]
    backgrounds = backgroundEstimator.estimateBatch(imageWithFaceDetectionList)
    pprint.pprint(backgrounds)


async def asyncEstimateBackground():
    """
    Example of an async background estimation.
    """
    image = VLImage.load(filename=EXAMPLE_4)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    backgroundEstimator = faceEngine.createFaceDetectionBackgroundEstimator()
    faceDetection = detector.detectOne(image)
    # async estimation
    imageWithFaceDetection = ImageWithFaceDetection(image, faceDetection.boundingBox)
    backgrounds = await backgroundEstimator.estimate(imageWithFaceDetection, asyncEstimate=True)
    pprint.pprint(backgrounds.asDict())
    # run tasks and get results
    task1 = backgroundEstimator.estimate(imageWithFaceDetection, asyncEstimate=True)
    task2 = backgroundEstimator.estimate(imageWithFaceDetection, asyncEstimate=True)
    for task in (task1, task2):
        pprint.pprint(task.get().asDict())


if __name__ == "__main__":
    estimateBackground()
    asyncio.run(asyncEstimateBackground())
