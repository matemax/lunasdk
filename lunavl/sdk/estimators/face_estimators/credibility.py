"""
Module for estimate a credibility person by face.
"""
from enum import Enum
from typing import Union

from FaceEngine import CredibilityCheckEstimation
from FaceEngine import CredibilityStatus
from FaceEngine import ICredibilityCheckEstimatorPtr

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, assertError
from ..base import BaseEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...async_task import AsyncTask
from ...base import BaseEstimation


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


def postProcessing(error, credibility):
    assertError(error)
    return Credibility(credibility)


class CredibilityEstimator(BaseEstimator):
    """
    Warp credibility estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, credibilityEstimator: ICredibilityCheckEstimatorPtr):
        """
        Init.
        Args:
            credibilityEstimator: core credibility check estimator
        """
        super().__init__(credibilityEstimator)

    @CoreExceptionWrap(LunaVLError.CredibilityError)
    def estimate(
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
            return AsyncTask(task, postProcessing)
        error, credibility = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        return postProcessing(error, credibility)
