"""
Module contains an orientation mode estimator.

See `orientation mode`_.
"""
from enum import Enum
from typing import Dict, Union

from FaceEngine import IOrientationEstimatorPtr, OrientationType
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, CoreExceptionWrap
from lunavl.sdk.base import BaseEstimation, BoundingBox
from lunavl.sdk.image_utils.image import VLImage
from .facewarper import FaceWarp, FaceWarpedImage

from ..base import BaseEstimator


class OrientationMode(BaseEstimation):
    """
    Orientation mode. Estimates the image orientation.
    """

    #  pylint: disable=W0235
    def __init__(self, coreOrientationMode: OrientationType):
        """
        Init.

        Args:
            coreOrientationMode: core orientation mode estimation.
        """

        super().__init__(coreOrientationMode)

    @property
    def orientationMode(self) -> str:
        """
        Get orientation type.

        Returns:
            one of
        """
        return self._coreEstimation.name

    def asDict(self) -> Dict[str, str]:
        """
        Convert angles to dict.

        Returns:
            {"pitch": self.pitch, "roll": self.roll, "yaw": self.yaw}
        """
        return {"orientation_mode": self.orientationMode}


class OrientationModeEstimator(BaseEstimator):
    """
    OrientationModeEstimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreOrientationModeEstimator: IOrientationEstimatorPtr):
        """
        Init.

        Args:
            coreOrientationModeEstimator: core orientation mode estimator
        """
        super().__init__(coreOrientationModeEstimator)

    @CoreExceptionWrap(LunaVLError.EstimationOrientationModeError)
    def estimate(self, warp: Union[FaceWarp, FaceWarpedImage]) -> OrientationMode:
        """
        Estimate orientation mode from warped image.

        Args:
            warp: warped image

        Returns:
            estimated orientation mode
        Raises:
            LunaSDKException: if estimation is failed
        """
        error, orientationModeEstimation = self._coreEstimator.estimate(warp.warpedImage.coreImage)

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

        return OrientationMode(orientationModeEstimation)
