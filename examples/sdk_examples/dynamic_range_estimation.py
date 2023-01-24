"""
Dynamic range estimation example
"""
import asyncio
import pprint

from resources import EXAMPLE_1, EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateDynamicRange():
    """
    Dynamic range estimation example.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)

    dynamicRangeEstimator = faceEngine.createDynamicRangeEstimator()

    pprint.pprint(dynamicRangeEstimator.estimate(faceDetection))

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetections = detector.detect([image, image2])

    estimations = dynamicRangeEstimator.estimateBatch([faceDetections[0][0], faceDetections[1][0]])
    pprint.pprint([estimation for estimation in estimations])


async def asyncEstimateDynamicRange():
    """
    Async dynamic range estimation example.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)

    dynamicRangeEstimator = faceEngine.createDynamicRangeEstimator()

    pprint.pprint(await dynamicRangeEstimator.estimate(faceDetection, asyncEstimate=True))

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetections = detector.detect([image, image2])

    estimations = await dynamicRangeEstimator.estimateBatch(
        [faceDetections[0][0], faceDetections[1][0]], asyncEstimate=True
    )
    pprint.pprint([estimation for estimation in estimations])


if __name__ == "__main__":
    estimateDynamicRange()
    asyncio.run(asyncEstimateDynamicRange())
