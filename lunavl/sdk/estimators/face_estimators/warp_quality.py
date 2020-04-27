"""Module for estimate a warped image quality.

See `warp quality`_.
"""
from typing import Dict, Union

from FaceEngine import SubjectiveQuality as CoreQuality, IQualityEstimatorPtr  # pylint: disable=E0611,E0401
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException

from lunavl.sdk.estimators.base_estimation import BaseEstimator
from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.estimators.face_estimators.warper import Warp, WarpedImage


class Quality(BaseEstimation):
    """
    Structure quality

    Estimation properties:

        - dark
        - blur
        - illumination
        - specularity
        - light
    """

    #  pylint: disable=W0235
    def __init__(self, coreQuality: CoreQuality):
        """
        Init.

        Args:
            coreQuality: estimated core quality
        """
        super().__init__(coreQuality)

    @property
    def blur(self) -> float:
        """
        Get blur.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.blur

    @property
    def dark(self) -> float:
        """
        Get dark.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.darkness

    @property
    def illumination(self) -> float:
        """
        Get illumination.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.illumination

    @property
    def specularity(self) -> float:
        """
        Get specularity.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.specularity

    @property
    def light(self) -> float:
        """
        Get light.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.light

    def asDict(self) -> Dict[str, float]:
        """
        Convert to dict.

        Returns:
            {"blurriness": self.blur, "dark": self.dark, "illumination": self.illumination,
             "specularity": self.specularity, "light": self.light}
        """
        return {
            "blurriness": self.blur,
            "dark": self.dark,
            "illumination": self.illumination,
            "specularity": self.specularity,
            "light": self.light,
        }


class WarpQualityEstimator(BaseEstimator):
    """
    Warp quality estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: IQualityEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core quality estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    @CoreExceptionWrap(LunaVLError.EstimationWarpQualityError)
    def estimate(self, warp: Union[Warp, WarpedImage]) -> Quality:
        """
        Estimate quality from a warp.

        Args:
            warp: raw warped image or warp

        Returns:
            estimated quality
        Raises:
            LunaSDKException: if estimation failed
        """
        error, coreQuality = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return Quality(coreQuality)
