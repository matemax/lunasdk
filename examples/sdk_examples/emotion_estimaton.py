"""
An emotion estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateEmotion():
    """
    Estimate emotion from a warped image.
    """
    image = VLImage.load(filename="C:/temp/test.jpg")
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createWarper()
    warp = warper.warp(faceDetection)

    emotionEstimator = faceEngine.createEmotionEstimator()

    pprint.pprint(emotionEstimator.estimate(warp.warpedImage).asDict())


if __name__ == "__main__":
    estimateEmotion()
