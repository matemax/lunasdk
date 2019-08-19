"""
Eyes estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateGazeDirection():
    """
    Estimate gaze direction.
    """
    image = VLImage.load(filename='C:/temp/test.jpg')
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image, detect68Landmarks=True)

    warper = faceEngine.createWarper()
    warp = warper.warp(faceDetection)
    landMarks5Transformation = warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")

    eyesEstimator = faceEngine.createEyeEstimator()

    eyesEstimation = eyesEstimator.estimate(landMarks5Transformation, warp.warpedImage)

    headPoseEstimator = faceEngine.createHeadPoseEstimator()
    faceDetection = detector.detectOne(image, detect5Landmarks=False, detect68Landmarks=True)
    #: estimate by 68 landmarks
    headPose = headPoseEstimator.estimateBy68Landmarks(faceDetection.landmarks68)

    gazeEstimator = faceEngine.createGazeEstimator()

    pprint.pprint(gazeEstimator.estimate(headPose, eyesEstimation).asDict())


if __name__ == "__main__":
    estimateGazeDirection()
