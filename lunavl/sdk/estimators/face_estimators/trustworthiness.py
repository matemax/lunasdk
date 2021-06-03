"""
Module for estimate a trustworthiness person by face.
"""
from typing import Union, Dict

from FaceEngine import ICredibilityCheckEstimatorPtr
from FaceEngine import CredibilityCheckEstimation

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException
from ..base import BaseEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...base import BaseEstimation


class Trustworthiness(BaseEstimation):
    """
    Structure trustworthiness estimation

    Estimation properties:

        - trustworthiness
    """

    #  pylint: disable=W0235
    def __init__(self, trustworthiness: CredibilityCheckEstimation):
        """
        Init.
        Args:
            trustworthiness: estimated trustworthiness
        """
        super().__init__(trustworthiness)

    @property
    def trustworthiness(self) -> float:
        """
        The trustworthiness

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.value

    def asDict(self) -> Dict[str, float]:
        """
        Convert to dict.

        Returns:
            self.trustworthiness
        }
        """
        return {"trustworthiness": self.trustworthiness}


class TrustworthinessEstimator(BaseEstimator):
    """
    Warp trustworthiness estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, trustworthinessEstimator: ICredibilityCheckEstimatorPtr):
        """
        Init.
        Args:
            trustworthinessEstimator: core credibility check estimator
        """
        super().__init__(trustworthinessEstimator)

    @CoreExceptionWrap(LunaVLError.TrustworthinessError)
    def estimate(self, warp: Union[FaceWarp, FaceWarpedImage]) -> Trustworthiness:
        """
        Estimate trustworthiness from a warp.

        Args:
            warp: raw warped image or warp

        Returns:
            estimated trustworthiness
        Raises:
            LunaSDKException: if estimation failed
        """
        error, trustworthiness = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return Trustworthiness(trustworthiness)
