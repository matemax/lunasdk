"""Module for estimate a warped image quality.

See `warp quality`_.
"""
from enum import Enum
from typing import Union, Dict

from FaceEngine import GlassesEstimation, IGlassesEstimatorPtr

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, assertError
from ..base import BaseEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...async_task import AsyncTask
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


def postProcessing(error, glasses) -> Glasses:
    assertError(error)
    return Glasses(glasses)


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

    @CoreExceptionWrap(LunaVLError.EstimationGlassesError)
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
            return AsyncTask(task, postProcessing)
        error, glasses = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        return postProcessing(error, glasses)
