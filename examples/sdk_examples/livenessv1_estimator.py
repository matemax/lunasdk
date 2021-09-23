"""
LivenessV1 estimation example
"""
import asyncio
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O, EXAMPLE_1


def estimateLiveness():
    """
    Estimate liveness.
    """

    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image, detect68Landmarks=True)

    livenessEstimator = faceEngine.createLivenessV1Estimator()

    pprint.pprint(livenessEstimator.estimate(faceDetection, qualityThreshold=0.5).asDict())

    faceDetection2 = detector.detectOne(VLImage.load(filename=EXAMPLE_1), detect68Landmarks=True)
    pprint.pprint(livenessEstimator.estimateBatch([faceDetection, faceDetection2]))


async def asyncEstimateLiveness():
    """
    Async estimate liveness.
    """

    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image, detect68Landmarks=True)

    livenessEstimator = faceEngine.createLivenessV1Estimator()

    liveness = await livenessEstimator.estimate(faceDetection, qualityThreshold=0.5, asyncEstimate=True)
    pprint.pprint(liveness.asDict())

    faceDetection2 = detector.detectOne(VLImage.load(filename=EXAMPLE_1), detect68Landmarks=True)
    task1 = livenessEstimator.estimateBatch([faceDetection, faceDetection], asyncEstimate=True)
    task2 = livenessEstimator.estimateBatch([faceDetection, faceDetection2], asyncEstimate=True)

    for task in (task1, task2):
        pprint.pprint(task.get())


if __name__ == "__main__":
    estimateLiveness()
    asyncio.run(asyncEstimateLiveness())
