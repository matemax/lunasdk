"""
Red-eye estimation example
"""
import pprint

from resources import EXAMPLE_1, EXAMPLE_O

from lunavl.sdk.estimators.face_estimators.eyes import WarpWithLandmarks5
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateRedEye():
    """
    Red-eye estimation example.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createFaceWarper()
    warp = warper.warp(faceDetection)
    landMarks5Transformation = warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")

    redEyeEstimator = faceEngine.createRedEyeEstimator()

    warpWithLandmarks = WarpWithLandmarks5(warp, landMarks5Transformation)
    pprint.pprint(redEyeEstimator.estimate(warpWithLandmarks).asDict())

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetection2 = detector.detectOne(image2)
    warp2 = warper.warp(faceDetection2)
    landMarks5Transformation2 = warper.makeWarpTransformationWithLandmarks(faceDetection2, "L5")

    warpWithLandmarksList = [
        WarpWithLandmarks5(warp, landMarks5Transformation),
        WarpWithLandmarks5(warp2, landMarks5Transformation2),
    ]

    estimations = redEyeEstimator.estimateBatch(warpWithLandmarksList)
    pprint.pprint([estimation.asDict() for estimation in estimations])


if __name__ == "__main__":
    estimateRedEye()
