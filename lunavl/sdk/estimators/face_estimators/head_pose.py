"""
Module contains a head pose estimator.

See `head pose`_.
"""
from enum import Enum
from typing import Dict, List

from FaceEngine import IHeadPoseEstimatorPtr, HeadPoseEstimation, FrontalFaceType  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, CoreExceptionWrap
from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.detectors.facedetector import Landmarks68
from ..base import BaseEstimator, ImageWithFaceDetection
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator


class FrontalType(Enum):
    """
    Enum for frontal types
    """

    TURNED = "Non"  #: Non-frontal face
    FRONTAL = "Good"  #: Good for recognition; Doesn't descrease recall and looks fine
    BY_GOST = "ISO"  #: GOST/ISO angles

    @classmethod
    def fromCoreFrontalType(cls, frontalFaceType: FrontalFaceType) -> "FrontalType":
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

    #  pylint: disable=W0235
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

    #  pylint: disable=W0235
    def __init__(self, coreHeadPoseEstimator: IHeadPoseEstimatorPtr):
        """
        Init.

        Args:
            coreHeadPoseEstimator: core estimator
        """
        super().__init__(coreHeadPoseEstimator)

    @CoreExceptionWrap(LunaVLError.EstimationHeadPoseError)
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
        error, headPoseEstimation = self._coreEstimator.estimate(landmarks68.coreEstimation)

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

        return HeadPose(headPoseEstimation)

    #  pylint: disable=W0221
    def estimate(self, landmarks68: Landmarks68) -> HeadPose:  # type: ignore
        """
        Realize interface of a abstract  estimator. Call estimateBy68Landmarks
        """
        return self.estimateBy68Landmarks(landmarks68)

    @CoreExceptionWrap(LunaVLError.EstimationHeadPoseError)
    def estimateByBoundingBox(self, imageWithFaceDetection: ImageWithFaceDetection) -> HeadPose:
        """
        Estimate head pose by detection.

        Args:
            imageWithFaceDetection: image with face detection
        Returns:
            estimate head pose
        Raises:
            LunaSDKException: if estimation is failed
        """
        error, headPoseEstimation = self._coreEstimator.estimate(
            imageWithFaceDetection.image.coreImage, imageWithFaceDetection.boundingBox.coreEstimation
        )

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return HeadPose(headPoseEstimation)

    @CoreExceptionWrap(LunaVLError.EstimationHeadPoseError)
    def estimateByBoundingBoxBatch(self, imageWithFaceDetectionList: List[ImageWithFaceDetection]) -> List[HeadPose]:
        """
        Batch estimate head pose by detection.

        Args:
            imageWithFaceDetectionList: list of image with face detection
        Returns:
            list of head pose estimations
        Raises:
            LunaSDKException: if estimation is failed
        """
        coreImages = [row.image.coreImage for row in imageWithFaceDetectionList]
        detections = [row.boundingBox.coreEstimation for row in imageWithFaceDetectionList]

        validateInputByBatchEstimator(self._coreEstimator, coreImages, detections)
        error, headPoseEstimations = self._coreEstimator.estimate(coreImages, detections)

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return [HeadPose(estimation) for estimation in headPoseEstimations]
