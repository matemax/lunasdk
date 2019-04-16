"""
Module realize simple examples following features:
    * head pose estimation
    * batch images face detection
    * detect landmarks68 and landmarks5
"""
import pprint

from lunavl.sdk.faceengine.engine import FACE_ENGINE
from lunavl.sdk.faceengine.facedetector import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateHeadPose():
    image = VLImage.load(
        url='https://cdn1.savepice.ru/uploads/2019/4/15/194734af15c4fcd06dec6db86bbeb7cd-full.jpg')
    detector = FACE_ENGINE.createFaceDetector(DetectorType.FACE_DET_V1)
    headPoseEstimator = FACE_ENGINE.createHeadPoseEstimator()
    faceDetection = detector.detectOne(image, detect5Landmarks=False, detect68Landmarks=True)
    angles = headPoseEstimator.estimate(faceDetection.landmarks68)
    angles.getFrontalFaceType()
    pprint.pprint(angles)


if __name__ == "__main__":
    estimateHeadPose()
