"""
Module contains a  livenessv1 estimator.

See `livenessv1`_.
"""
from enum import Enum
from typing import List, Literal, Optional, Union, overload

from FaceEngine import LivenessOneShotRGBEstimation  # pylint: disable=E0611,E0401
from FaceEngine import FSDKErrorResult, LivenessOneShotState

from lunavl.sdk.async_task import AsyncTask
from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.errors.exceptions import assertError
from lunavl.sdk.estimators.base import BaseEstimator
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator


class LivenessPrediction(Enum):
    """
    Liveness estimation prediction
    """

    Real = "real"  # real human
    Spoof = "spoof"  # spoof
    Unknown = "unknown"  # unknown

    @staticmethod
    def fromCoreEmotion(coreState: LivenessOneShotState) -> "LivenessPrediction":
        """
        Get enum element by core liveness.

        Args:
            coreState: enum value from core

        Returns:
            corresponding prediction
        """
        if coreState == LivenessOneShotState.Alive:
            return LivenessPrediction.Real
        if coreState == LivenessOneShotState.Fake:
            return LivenessPrediction.Spoof
        if coreState == LivenessOneShotState.Unknown:
            return LivenessPrediction.Unknown
        raise RuntimeError(f"bad core liveness state {coreState}")


class LivenessV1(BaseEstimation):
    """
    Liveness structure (LivenessOneShotRGBEstimation).

    Attributes:
        prediction: liveness prediction

    Estimation properties:

        - score
        - quality
    """

    __slots__ = ("prediction",)

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: LivenessOneShotRGBEstimation, prediction: LivenessPrediction):
        """
        Init.

        Args:
            coreEstimation: core estimation
        """
        self.prediction = prediction
        super().__init__(coreEstimation)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            {"prediction": self.prediction, "estimations": {"quality": self.quality, "score": self.score}}
        """
        return {"prediction": self.prediction.value, "estimations": {"quality": self.quality, "score": self.score}}

    @property
    def score(self) -> float:
        """
        Liveness score

        Returns:
            liveness score
        """
        return self._coreEstimation.score

    @property
    def quality(self) -> float:
        """
        Liveness quality score

        Returns:
            liveness quality score
        """
        return self._coreEstimation.qualityScore


def postProcessing(error: FSDKErrorResult, estimation: LivenessOneShotRGBEstimation) -> LivenessV1:
    """
    Convert a livenessV1 estimation from core result  and check error.
    Args:
        error: error
        estimations:  core batch livenesv1 estimation

    Returns:
        livenessv1
    """
    assertError(error)
    return LivenessV1(estimation, LivenessPrediction.fromCoreEmotion(estimation.State))


def postProcessingBatch(error: FSDKErrorResult, estimations: List[LivenessOneShotRGBEstimation]) -> List[LivenessV1]:
    """
    Convert a livenessV1 batch estimation from core result  and check error.
    Args:
        error: error
        estimations:  core batch livenesv1 estimation

    Returns:
        livenessv1 list
    """
    assertError(error)

    return [LivenessV1(estimation, LivenessPrediction.fromCoreEmotion(estimation.State)) for estimation in estimations]


class LivenessV1Estimator(BaseEstimator):
    """
    Liveness estimator version 1 (LivenessOneShotRGBEstimator).
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self,
        faceDetection: FaceDetection,
        qualityThreshold: Optional[float] = None,
        asyncEstimate: Literal[False] = False,
    ) -> LivenessV1:
        ...

    @overload
    def estimate(
        self,
        faceDetection: FaceDetection,
        qualityThreshold: Optional[float] = None,
        asyncEstimate: Literal[True],
    ) -> AsyncTask[LivenessV1]:
        ...

    def estimate(  # type: ignore
        self,
        faceDetection: FaceDetection,
        qualityThreshold: Optional[float] = None,
        asyncEstimate: bool = False,
    ) -> Union[LivenessV1, AsyncTask[LivenessV1]]:
        """
        Estimate a liveness

        .. warning::
            Current estimator version estimates correct liveness state for images from mobile and web camera only.
            A correctness of a liveness prediction are not guarantee for other images source.

        Args:
            faceDetection: face detection
            qualityThreshold: quality threshold. if estimation quality is low of this threshold
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated liveness if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if faceDetection.landmarks5 is None:
            raise ValueError("Landmarks5 is required for liveness estimation")
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(
                faceDetection.image.coreImage,
                faceDetection.coreEstimation.detection,
                faceDetection.landmarks5.coreEstimation,
                -1.0 if qualityThreshold is None else qualityThreshold,
            )
            return AsyncTask(task, postProcessing)
        error, estimation = self._coreEstimator.estimate(
            faceDetection.image.coreImage,
            faceDetection.coreEstimation.detection,
            faceDetection.landmarks5.coreEstimation,
            -1.0 if qualityThreshold is None else qualityThreshold,
        )
        return postProcessing(error, estimation)

    #  pylint: disable=W0221
    def estimateBatch(  # type: ignore
        self,
        faceDetections: List[FaceDetection],
        qualityThreshold: Optional[float] = None,
        asyncEstimate: bool = False,
    ) -> Union[List[LivenessV1], AsyncTask[List[LivenessV1]]]:
        """
        Batch estimate liveness

        .. warning::
            Current estimator version estimates correct liveness state for images from mobile and web camera only.
            A correctness of the liveness prediction is not guaranteed for other image sources.

        Args:
            faceDetections: face detection list
            qualityThreshold: quality threshold. if estimation quality is low of this threshold
            asyncEstimate: estimate or run estimation in background
        Returns:
            estimated liveness if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        coreImages = [detection.image.coreImage for detection in faceDetections]
        detections = [detection.coreEstimation.detection for detection in faceDetections]
        try:
            coreEstimations = [detection.landmarks5.coreEstimation for detection in faceDetections]  # type: ignore
        except AttributeError:
            raise ValueError("Landmarks5 is required for liveness estimation")

        validateInputByBatchEstimator(self._coreEstimator, coreImages, detections, coreEstimations)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(
                coreImages,
                detections,
                coreEstimations,
                -1.0 if qualityThreshold is None else qualityThreshold,
            )
            return AsyncTask(task, postProcessingBatch)
        error, estimations = self._coreEstimator.estimate(
            coreImages,
            detections,
            coreEstimations,
            -1.0 if qualityThreshold is None else qualityThreshold,
        )
        return postProcessingBatch(error, estimations)
