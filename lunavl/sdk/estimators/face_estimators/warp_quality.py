"""Module for estimate a warped image quality.

See `warp quality`_.
"""
from typing import Dict, Union, List

from FaceEngine import SubjectiveQuality as CoreQuality, IQualityEstimatorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, assertError
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...async_task import AsyncTask


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


def postProcessing(error, quality):
    assertError(error)
    return Quality(quality)


def postProcessingBatch(error, qualities):
    assertError(error)

    return [Quality(quality) for quality in qualities]


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
    def estimate(
        self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: bool = False
    ) -> Union[Quality, AsyncTask[Quality]]:
        """
        Estimate quality from a warp.

        Args:
            warp: raw warped image or warp
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated quality if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(warp.warpedImage.coreImage)
            return AsyncTask(task, postProcessing)
        error, emotions = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        return postProcessing(error, emotions)

    #  pylint: disable=W0221
    @CoreExceptionWrap(LunaVLError.EstimationWarpQualityError)
    def estimateBatch(
        self, warps: List[Union[FaceWarp, FaceWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[Quality], AsyncTask[List[Quality]]]:
        """
        Batch estimate emotions

        Args:
            warps: warped images
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated quality if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        coreImages = [warp.warpedImage.coreImage for warp in warps]

        validateInputByBatchEstimator(self._coreEstimator, coreImages)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, postProcessingBatch)
        error, masks = self._coreEstimator.estimate(coreImages)
        return postProcessingBatch(error, masks)
