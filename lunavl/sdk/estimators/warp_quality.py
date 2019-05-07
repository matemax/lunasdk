"""Module for estimate a warped image quality.
"""
from typing import Dict, Union

from FaceEngine import Quality as CoreQuality, IQualityEstimatorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.estimators.base_estimation import BaseEstimation, BaseEstimator
from lunavl.sdk.faceengine.warper import Warp, WarpedImage


class Quality(BaseEstimation):
    """
    Structure quality

    Estimation properties:

        - dark
        - blur
        - gray
        - light
    """

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
        return self._coreEstimation.dark

    @property
    def gray(self) -> float:
        """
        Get gray.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.gray

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
            {"darkness": self.dark, "lightning": self.light, "saturation": self.gray, "blurness": self.blur}
        """
        return {"darkness": self.dark, "lightning": self.light, "saturation": self.gray, "blurness": self.blur}


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
    def estimate(self, warp: Union[Warp, WarpedImage]) -> Quality:
        """
        Estimate quality from a warp.

        Args:
            warp: raw warped image or warp

        Returns:
            estimated quality
        """
        error, coreQuality = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        if error.isError:
            raise ValueError("1234yui")
        return Quality(coreQuality)
