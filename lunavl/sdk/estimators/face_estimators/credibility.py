"""
Module for estimate a credibility person by face.
"""
from enum import Enum
from typing import Union

from FaceEngine import CredibilityCheckEstimation
from FaceEngine import CredibilityStatus
from FaceEngine import ICredibilityCheckEstimatorPtr

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException
from ..base import BaseEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
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
    def estimate(self, warp: Union[FaceWarp, FaceWarpedImage]) -> Credibility:
        """
        Estimate credibility from a warp.

        Args:
            warp: raw warped image or warp

        Returns:
            estimated credibility
        Raises:
            LunaSDKException: if estimation failed
        """
        error, credibility = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return Credibility(credibility)
