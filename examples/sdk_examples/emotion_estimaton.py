"""
An emotion estimation example
"""
import asyncio
import pprint

from resources import EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateEmotion():
    """
    Estimate emotion from a warped image.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)

    emotionEstimator = faceEngine.createEmotionEstimator()

    pprint.pprint(emotionEstimator.estimate(warp.warpedImage).asDict())


async def asyncEstimateEmotion():
    """
    Async estimate emotion from a warped image.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)

    emotionEstimator = faceEngine.createEmotionEstimator()

    emotions = await emotionEstimator.estimate(warp.warpedImage, asyncEstimate=True)
    pprint.pprint(emotions.asDict())
    task = emotionEstimator.estimate(warp.warpedImage, asyncEstimate=True)
    pprint.pprint(task.get().asDict())


if __name__ == "__main__":
    estimateEmotion()
    asyncio.run(asyncEstimateEmotion())
