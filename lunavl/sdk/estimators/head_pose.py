"""
Module contains a head pose estimator
"""
from enum import Enum
from typing import Dict

from FaceEngine import IHeadPoseEstimatorPtr, HeadPoseEstimation, FrontalFaceType  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import ErrorInfo
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.base_estimation import BaseEstimation, BaseEstimator
from lunavl.sdk.faceengine.facedetector import Landmarks68, BoundingBox
from lunavl.sdk.image_utils.image import VLImage


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

    def __repr__(self):
        return self.value


class HeadPose(BaseEstimation):
    """
    Head pose. Estimate Tait–Bryan angles for head (https://en.wikipedia.org/wiki/Euler_angles#Tait–Bryan_angles).
    Estimation properties:

        - pitch
        - roll
        - yaw
    """

    def __init__(self, coreHeadPose: HeadPoseEstimation):
        """
        Init.

        Args:
            coreHeadPose: core head pose estimation.
        """

        super().__init__(coreHeadPose)

    @property
    def yaw(self) -> float:
        """
        Get the yaw angle.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.yaw

    @property
    def pitch(self) -> float:
        """
        Get the pitch angle.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.pitch

    @property
    def roll(self) -> float:
        """
        Get the pitch angle.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.roll

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
        return FrontalType.fromCoreFrontalType(self._coreEstimation.getFrontalFaceType())


class HeadPoseEstimator(BaseEstimator):
    """
    HeadPoseEstimator.
    """

    def __init__(self, coreHeadPoseEstimator: IHeadPoseEstimatorPtr):
        """
        Init.

        Args:
            coreHeadPoseEstimator: core estimator
        """
        super().__init__(coreHeadPoseEstimator)

    def estimateBy68Landmarks(self, landmarks68: Landmarks68) -> HeadPose:
        """
        Estimate head pose by 68 landmarks.

        Args:
            landmarks68: landmarks68

        Returns:
            estimate head pose
        Raises:
            LunaSDKException: if estimation is failed
        """
        err, headPoseEstimation = self._coreEstimator.estimate(landmarks68.coreEstimation)

        if err.isError:
            error = ErrorInfo.fromSDKError(125, "head pose estimation", err)
            raise LunaSDKException(error)
        return HeadPose(headPoseEstimation)

    def estimate(self,  landmarks68: Landmarks68) -> HeadPose:
        """
        Realize interface of a abstract  ectimator. Call estimateBy68Landmarks
        """
        return self.estimateBy68Landmarks(landmarks68)

    def estimateByBoundingBox(self, detection: BoundingBox, imageWithDetection: VLImage) -> HeadPose:
        """
        Estimate head pose by detection.

        Args:
            detection: detection bounding box
            imageWithDetection: image with the detection.
        Returns:
            estimate head pose
        Raises:
            LunaSDKException: if estimation is failed
        """
        err, headPoseEstimation = self._coreEstimator.estimate(imageWithDetection.coreImage,
                                                                       detection.coreEstimation)

        if err.isError:
            error = ErrorInfo.fromSDKError(125, "head pose estimation", err)
            raise LunaSDKException(error)
        return HeadPose(headPoseEstimation)
