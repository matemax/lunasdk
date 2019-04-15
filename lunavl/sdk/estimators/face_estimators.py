from FaceEngine import IHeadPoseEstimatorPtr

from lunavl.sdk.faceengine.facedetector import Landmarks68


class HeadPose:
    def __init__(self, coreHeadPose):
        self.pith = 1
        self.yaw = 2
        self.roll = 3


class HeadPoseEstimator:

    def __init__(self, coreHeadPoseEstimator: IHeadPoseEstimatorPtr):
        self._coreHeadPoseEstimator = coreHeadPoseEstimator

    def estimate(self, landmrks68: Landmarks68):
        err, headPoseEstimation = self._coreHeadPoseEstimator(landmrks68.coreLandmarks)
        return HeadPose(headPoseEstimation)
