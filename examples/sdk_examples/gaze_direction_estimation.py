"""
Eyes estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import FACE_ENGINE
from lunavl.sdk.faceengine.facedetector import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateGazeDirection():
    """
    Estimate gaze direction.
    """
    image = VLImage.load(filename='C:/temp/test.jpg')
    detector = FACE_ENGINE.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image, detect68Landmarks=True)

    warper = FACE_ENGINE.createWarper()
    warp = warper.warp(faceDetection)
    landMarks5Transformation = warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")

    eyesEstimator = FACE_ENGINE.createEyeEstimator()

    eyesEstimation = eyesEstimator.estimate(landMarks5Transformation, warp.warpedImage)

    headPoseEstimator = FACE_ENGINE.createHeadPoseEstimator()
    faceDetection = detector.detectOne(image, detect5Landmarks=False, detect68Landmarks=True)
    #: estimate by 68 landmarks
    headPose = headPoseEstimator.estimateBy68Landmarks(faceDetection.landmarks68)

    gazeEstimator = FACE_ENGINE.createGazeEstimator()

    pprint.pprint(gazeEstimator.estimate(headPose, eyesEstimation))


if __name__ == "__main__":
    estimateGazeDirection()
