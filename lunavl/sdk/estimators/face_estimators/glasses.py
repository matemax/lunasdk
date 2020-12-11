"""Module for estimate a warped image quality.

See `warp quality`_.
"""
from typing import Union, Dict
from enum import Enum

from FaceEngine import GlassesEstimation, IGlassesEstimatorPtr

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException
from ..base import BaseEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
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

    def fromCoreGlasses(coreGlassesName: str) -> "GlassesState":
        if coreGlassesName == 'NoGlasses':
            return GlassesState.NoGlasses
        if coreGlassesName == 'EyeGlasses':
            return GlassesState.EyeGlasses
        if coreGlassesName == 'SunGlasses':
            return GlassesState.SunGlasses
        raise RuntimeError(f"bad core glasses state {coreGlassesName}")


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
        return {
            "glasses": self.glasses.name.lower() if self.glasses.name != 'NoGlasses' else 'no_glasses',
        }


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
    def estimate(self, warp: Union[FaceWarp, FaceWarpedImage]) -> Glasses:
        """
        Estimate glasses from a warp.

        Args:
            warp: raw warped image or warp

        Returns:
            estimated glasses
        Raises:
            LunaSDKException: if estimation failed
        """
        error, glasses = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return Glasses(glasses)
