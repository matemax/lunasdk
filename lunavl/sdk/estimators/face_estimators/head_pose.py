"""
Module contains a head pose estimator.

See `head pose`_.
"""
from enum import Enum
from typing import Dict, List, Literal, Union, overload

from FaceEngine import FrontalFaceType, HeadPoseEstimation  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.detectors.facedetector import FaceDetection, Landmarks68

from ...async_task import AsyncTask, DefaultPostprocessingFactory
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


POST_PROCESSING = DefaultPostprocessingFactory(HeadPose)


class HeadPoseEstimator(BaseEstimator):
    """
    HeadPoseEstimator.
    """

    def estimateBy68Landmarks(
        self, landmarks68: Landmarks68, asyncEstimate: bool = False
    ) -> Union[HeadPose, AsyncTask[HeadPose]]:
        """
        Estimate head pose by 68 landmarks.

        Args:
            landmarks68: landmarks68
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimate head pose if asyncExecute is False otherwise async task
        Raises:
            LunaSDKException: if estimation is failed
        """
        if not asyncEstimate:
            error, headPoseEstimation = self._coreEstimator.estimate(landmarks68.coreEstimation)
            return POST_PROCESSING.postProcessing(error, headPoseEstimation)
        task = self._coreEstimator.asyncEstimate(landmarks68.coreEstimation)
        return AsyncTask(task, POST_PROCESSING.postProcessing)

    #  pylint: disable=W0221
    def estimate(  # type: ignore
        self, landmarks68: Landmarks68, asyncEstimate: bool = False
    ) -> Union[HeadPose, AsyncTask[HeadPose]]:
        """
        Realize interface of a abstract  estimator. Call estimateBy68Landmarks
        """
        return self.estimateBy68Landmarks(landmarks68, asyncEstimate=asyncEstimate)

    @overload
    def estimateByBoundingBox(
        self, imageWithFaceDetection: ImageWithFaceDetection, asyncEstimate: Literal[False] = False
    ) -> HeadPose:
        ...

    @overload
    def estimateByBoundingBox(
        self, imageWithFaceDetection: ImageWithFaceDetection, asyncEstimate: Literal[True]
    ) -> AsyncTask[HeadPose]:
        ...

    def estimateByBoundingBox(
        self, imageWithFaceDetection: ImageWithFaceDetection, asyncEstimate: bool = False
    ) -> Union[HeadPose, AsyncTask[HeadPose]]:
        """
        Estimate head pose by detection.

        Args:
            imageWithFaceDetection: image with face detection
            asyncEstimate: estimate or run estimation in background
        Returns:
            head pose estimation if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation is failed
        """
        if not asyncEstimate:
            error, headPoseEstimation = self._coreEstimator.estimate(
                imageWithFaceDetection.image.coreImage, imageWithFaceDetection.boundingBox.coreEstimation
            )
            return POST_PROCESSING.postProcessing(error, headPoseEstimation)
        task = self._coreEstimator.asyncEstimate(
            imageWithFaceDetection.image.coreImage, imageWithFaceDetection.boundingBox.coreEstimation
        )
        return AsyncTask(task, POST_PROCESSING.postProcessing)

    def estimateBatch(
        self, batch: Union[List[ImageWithFaceDetection], List[FaceDetection]], asyncEstimate: bool = False
    ) -> Union[List[HeadPose], AsyncTask[List[HeadPose]]]:
        """
        Batch estimate head pose by detection.

        Args:
            batch: list of image with face detection or face detections
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of head pose estimations if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation is failed
        """

        coreImages = [row.image.coreImage for row in batch]
        detections = [row.boundingBox.coreEstimation for row in batch]

        validateInputByBatchEstimator(self._coreEstimator, coreImages, detections)
        if not asyncEstimate:
            error, headPoseEstimations = self._coreEstimator.estimate(coreImages, detections)
            return POST_PROCESSING.postProcessingBatch(error, headPoseEstimations)
        task = self._coreEstimator.asyncEstimate(coreImages, detections)
        return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
