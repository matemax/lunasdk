"""
Module contains utils for make face estimations
"""
from enum import Enum

from FaceEngine import IHeadPoseEstimatorPtr, HeadPoseEstimation, FrontalFaceType # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import ErrorInfo
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.faceengine.facedetector import Landmarks68


class FrontalType(Enum):
    """
    Enum for frontal types
    """
    TURNED = "FrontalFace0"  #: Non-frontal face
    FRONTAL = "FrontalFace1"  #: Good for recognition; Doesn't descrease recall and looks fine
    BY_GOST = "FrontalFace2"  #: GOST/ISO angles

    @classmethod
    def fromCoreFrontalType(cls, frontalFaceType: FrontalFaceType) -> 'FrontalType':
        """
        Create frontal type by core frontal type

        Args:
            frontalFaceType: core frontal type

        Returns:
            frontal type
        """
        frontalType = cls(frontalFaceType.name)
        return frontalType


class HeadPose:
    """
    Head pose. Aircraft principal axes for face https://en.wikipedia.org/wiki/Aircraft_principal_axes.

    Attributes:
        pitch (float): pitch
        yaw (float): pitch
        roll (float): pitch
    """
    __slots__ = ["pitch", "yaw", "roll", "_coreEstimation"]

    def __init__(self, coreHeadPose: HeadPoseEstimation):
        self.pitch = coreHeadPose.pitch
        self.yaw = coreHeadPose.yaw
        self.roll = coreHeadPose.roll
        self._coreEstimation = coreHeadPose

    def asDict(self):
        return {"pitch": self.pitch, "roll": self.roll, "yaw": self.yaw}

    def getFrontalFaceType(self) -> FrontalType:
        return FrontalType.fromCoreFrontalType(self._coreEstimation.getFrontalFaceType())

    def __repr__(self):
        return "pitch = {}, roll = {}, yaw = {}".format(self.pitch, self.roll, self.yaw)


class HeadPoseEstimator:

    def __init__(self, coreHeadPoseEstimator: IHeadPoseEstimatorPtr):
        self._coreHeadPoseEstimator = coreHeadPoseEstimator

    def estimate(self, landmrks68: Landmarks68):
        err, headPoseEstimation = self._coreHeadPoseEstimator.estimate(landmrks68.coreLandmarks)

        if err.isError:
            error = ErrorInfo.fromSDKError(125, "head pose estimation", err)
            raise LunaSDKException(error)
        return HeadPose(headPoseEstimation)
