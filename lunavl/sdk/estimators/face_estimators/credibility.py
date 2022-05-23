"""
Module for estimate a credibility person by face.
"""
from enum import Enum
from typing import Literal, Union, overload

from FaceEngine import CredibilityCheckEstimation, CredibilityStatus

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ...base import BaseEstimation
from ..base import BaseEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage


class CredibilityPrediction(Enum):
    """
    Credibility estimation prediction
    """

    Reliable = "reliable"  # human is reliable
    NonReliable = "non_reliable"  # human is not reliable

    @staticmethod
    def fromCoreEmotion(coreStatus: CredibilityStatus) -> "CredibilityPrediction":
        """
        Get enum element by core credibility status.

        Args:
            coreStatus: enum value from core

        Returns:
            corresponding prediction
        """
        if coreStatus == CredibilityStatus.Reliable:
            return CredibilityPrediction.Reliable
        if coreStatus == CredibilityStatus.NonReliable:
            return CredibilityPrediction.NonReliable
        raise RuntimeError(f"bad core credibility status {coreStatus}")


class Credibility(BaseEstimation):
    """
    Structure credibility estimation

    Estimation properties:

        - credibility
    """

    #  pylint: disable=W0235
    def __init__(self, credibility: CredibilityCheckEstimation):
        super().__init__(credibility)

    @property
    def score(self) -> float:
        """
        The credibility score

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.value

    @property
    def prediction(self) -> CredibilityPrediction:
        """
        Credibility prediction.

        Returns:
            get credibility prediction.
        """
        return CredibilityPrediction.fromCoreEmotion(self._coreEstimation.credibilityStatus)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            core credibility status
        """
        return {"estimations": {"score": self.score}, "prediction": self.prediction.value}


POST_PROCESSING = DefaultPostprocessingFactory(Credibility)


class CredibilityEstimator(BaseEstimator):
    """
    Warp credibility estimator.
    """

    @overload  # type: ignore
    def estimate(self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: Literal[False] = False) -> Credibility:
        ...

    @overload
    def estimate(self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: Literal[True]) -> AsyncTask[Credibility]:
        ...

    def estimate(  # type: ignore
        self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: bool = False
    ) -> Union[Credibility, AsyncTask[Credibility]]:
        """
        Estimate credibility from a warp.

        Args:
            warp: raw warped image or warp
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated credibility if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(warp.warpedImage.coreImage)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, credibility = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        return POST_PROCESSING.postProcessing(error, credibility)
