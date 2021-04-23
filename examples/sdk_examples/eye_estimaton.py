"""
Eyes estimation example
"""
import pprint

from lunavl.sdk.estimators.face_estimators.eyes import WarpWithLandmarks
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O, EXAMPLE_1


def estimateEyes():
    """
    Estimate emotion from a warped image.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    landMarks5Transformation = warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")

    eyesEstimator = faceEngine.createEyeEstimator()

    warpWithLandmarks = WarpWithLandmarks(landMarks5Transformation, warp.warpedImage.coreImage)
    pprint.pprint(eyesEstimator.estimate(warpWithLandmarks).asDict())

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetection2 = detector.detectOne(image2)
    warp2 = warper.warp(faceDetection2)
    landMarks5Transformation2 = warper.makeWarpTransformationWithLandmarks(faceDetection2, "L5")

    warpWithLandmarksList = [
        WarpWithLandmarks(landMarks5Transformation, warp.warpedImage.coreImage),
        WarpWithLandmarks(landMarks5Transformation2, warp2.warpedImage.coreImage),
    ]

    estimations = eyesEstimator.estimateBatch(warpWithLandmarksList)
    pprint.pprint([estimation.asDict() for estimation in estimations])


if __name__ == "__main__":
    estimateEyes()
