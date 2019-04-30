"""
Module for estimate a warped image quality.
"""
from typing import Dict, Union

from FaceEngine import Quality as CoreQuality, IQualityEstimatorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.faceengine.warper import Warp, WarpedImage


class Quality:
    """
    Structure quality

    Attributes:
        _coreQuality:
    """
    __slots__ = ["_coreQuality"]

    def __init__(self, coreQuality: CoreQuality):
        """
        Init.

        Args:
            coreQuality: estimated core quality
        """
        self._coreQuality = coreQuality

    @property
    def blur(self) -> float:
        """
        Get blur.

        Returns:
            float in range(0, 1)
        """
        return self._coreQuality.blur

    @property
    def dark(self) -> float:
        """
        Get dark.

        Returns:
            float in range(0, 1)
        """
        return self._coreQuality.dark

    @property
    def gray(self) -> float:
        """
        Get gray.

        Returns:
            float in range(0, 1)
        """
        return self._coreQuality.gray

    @property
    def light(self) -> float:
        """
        Get light.

        Returns:
            float in range(0, 1)
        """
        return self._coreQuality.light

    @property
    def coreQuality(self) -> CoreQuality:
        """
        Get estimated core quality.

        Returns:
            core quality
        """
        return self._coreQuality

    def asDict(self) -> Dict[str, float]:
        """
        Convert to dict.

        Returns:
            {"darkness": self.dark, "lightning": self.light, "saturation": self.gray, "blurness": self.blur}
        """
        return {"darkness": self.dark, "lightning": self.light, "saturation": self.gray, "blurness": self.blur}

    def __repr__(self) -> str:
        """
        Representation

        Returns:
            str(self.asDict())
        """
        return str(self.asDict())


class WarpQualityEstimator:
    """
    Warp quality estimator.

    Attributes:
        _coreQualityEstimator (IQualityEstimatorPtr):  core quality estimator
    """
    __slots__ = ["_coreQualityEstimator"]

    def __init__(self, coreEstimator: IQualityEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core quality estimator
        """
        self._coreQualityEstimator = coreEstimator

    def estimate(self, warp: Union[Warp, WarpedImage]) -> Quality:
        """
        Estimate quality from a warp.

        Args:
            warp: raw warped image or warp

        Returns:
            estimated quality
        """
        error, coreQuality = self._coreQualityEstimator.estimate(warp.warpedImage.coreImage)
        if error.isError:
            raise ValueError("1234yui")
        return Quality(coreQuality)
