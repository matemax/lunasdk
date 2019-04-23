"""
Module contains utils for make face estimations
"""
from enum import Enum
from typing import Dict

from FaceEngine import IHeadPoseEstimatorPtr, HeadPoseEstimation, FrontalFaceType  # pylint: disable=E0611,E0401

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
    Head pose. Estimate Tait–Bryan angles for head (https://en.wikipedia.org/wiki/Euler_angles#Tait–Bryan_angles).

    Attributes:
        pitch (float): pitch
        yaw (float): pitch
        roll (float): pitch
    """
    __slots__ = ["pitch", "yaw", "roll", "_coreEstimation"]

    def __init__(self, coreHeadPose: HeadPoseEstimation):
        """
        Init.

        Args:
            coreHeadPose: core head pose estimation.
        """
        self.pitch = coreHeadPose.pitch
        self.yaw = coreHeadPose.yaw
        self.roll = coreHeadPose.roll
        self._coreEstimation = coreHeadPose

    def asDict(self) -> Dict[str, float]:
        """
        Convert angles to dict.

        Returns:
            {"pitch": self.pitch, "roll": self.roll, "yaw": self.yaw}
        """
        return {"pitch": self.pitch, "roll": self.roll, "yaw": self.yaw}

    def getFrontalType(self) -> FrontalType:
        """
        Get frontal type of head pose estimation.

        Returns:
            frontal type
        """
        return FrontalType.fromCoreFrontalType(self._coreEstimation.getFrontalType())

    def __repr__(self) -> str:
        """
        Generate representation.

        Returns:
            "pitch = {self.pitch}, roll = {self.roll}, yaw = {self.yaw}"
        """
        return "pitch = {}, roll = {}, yaw = {}".format(self.pitch, self.roll, self.yaw)


class HeadPoseEstimator:
    """
    HeadPoseEstimator.
    Attributes
        _coreHeadPoseEstimator (IHeadPoseEstimatorPtr): core estimator.
    """
    __slots__ = ["_coreHeadPoseEstimator"]

    def __init__(self, coreHeadPoseEstimator: IHeadPoseEstimatorPtr):
        """
        Init.

        Args:
            coreHeadPoseEstimator: core estimator
        """
        self._coreHeadPoseEstimator = coreHeadPoseEstimator

    def estimate(self, landmarks68: Landmarks68) -> HeadPose:
        """
        Estimate head pose.

        Args:
            landmarks68: landmarks68

        Returns:
            estimate head pose
        Raises:
            LunaSDKException: if estimation is failed
        """
        err, headPoseEstimation = self._coreHeadPoseEstimator.estimate(landmarks68.coreLandmarks)

        if err.isError:
            error = ErrorInfo.fromSDKError(125, "head pose estimation", err)
            raise LunaSDKException(error)
        return HeadPose(headPoseEstimation)
