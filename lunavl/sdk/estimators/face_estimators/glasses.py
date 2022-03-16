"""Module for estimate a warped image quality.

See `warp quality`_.
"""
from enum import Enum
from typing import Union, Dict, List

from FaceEngine import GlassesEstimation, IGlassesEstimatorPtr

from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ...base import BaseEstimation


class GlassesState(Enum):
    """
    Glasses enum
    """

    #: NoGlasses
    NoGlasses = 1
    #: EyeGlasses
    EyeGlasses = 2
    #: SunGlasses
    SunGlasses = 3

    @staticmethod
    def fromCoreGlasses(coreGlassesName: str) -> "GlassesState":
        if coreGlassesName == "NoGlasses":
            return GlassesState.NoGlasses
        if coreGlassesName == "EyeGlasses":
            return GlassesState.EyeGlasses
        if coreGlassesName == "SunGlasses":
            return GlassesState.SunGlasses
        raise RuntimeError(f"bad core glasses state {coreGlassesName}")

    def __str__(self):
        return self.name.lower() if self.name != "NoGlasses" else "no_glasses"


class Glasses(BaseEstimation):
    """
    Structure glasses

    Estimation properties:

        - glasses
    """

    def __init__(self, glasses: GlassesEstimation):
        """
        Init.

        Args:
            glasses: estimated glasses
        """
        super().__init__(glasses)

    @property
    def glasses(self) -> GlassesState:
        """
        Get glasses state

        Returns:
            Glasses state with max score value
        """
        return GlassesState.fromCoreGlasses(self._coreEstimation.name)

    def asDict(self) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Convert to dict.

        Returns:
            {
                "glasses": predominantName,
            }

        """
        return {"glasses": str(self.glasses)}


POST_PROCESSING = DefaultPostprocessingFactory(Glasses)


class GlassesEstimator(BaseEstimator):
    """
    Warp glasses estimator.
    """

    def __init__(self, glassesEstimator: IGlassesEstimatorPtr):
        """
        Init.

        Args:
            glassesEstimator: core glasses estimator
        """
        super().__init__(glassesEstimator)

    def estimate(
        self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: bool = False
    ) -> Union[Glasses, AsyncTask[Glasses]]:
        """
        Estimate glasses from a warp.

        Args:
            warp: raw warped image or warp
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated glasses
        Raises:
            LunaSDKException: if estimation failed if asyncEstimate is false otherwise async task
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(warp.warpedImage.coreImage)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, glasses = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        return POST_PROCESSING.postProcessing(error, glasses)

    #  pylint: disable=W0221
    def estimateBatch(
        self, warps: List[Union[FaceWarp, FaceWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[Glasses], AsyncTask[List[Glasses]]]:
        """
        Batch estimate glasses

        Args:
            warps:warped images
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated glasses if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        coreImages = [warp.warpedImage.coreImage for warp in warps]

        validateInputByBatchEstimator(self._coreEstimator, coreImages)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        error, masks = self._coreEstimator.estimate(coreImages)
        return POST_PROCESSING.postProcessingBatch(error, masks)
