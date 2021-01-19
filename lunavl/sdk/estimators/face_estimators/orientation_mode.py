"""
Module contains an orientation mode estimator.

See `orientation mode`_.
"""
from enum import Enum
from typing import Dict

from FaceEngine import IOrientationEstimatorPtr, OrientationType as CoreOrientationType, Image as CoreImage

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, CoreExceptionWrap
from lunavl.sdk.base import BaseEstimation
from ..base import BaseEstimator


class OrientationType(Enum):
    """
    Enum for orientation type
    """

    LEFT = "Left"
    NORMAL = "Normal"
    RIGHT = "Right"
    UPSIDE_DOWN = "UpsideDown"

    @classmethod
    def fromCoreOrientationType(cls, coreOrientationMode: CoreOrientationType) -> "OrientationType":
        """
        Create orientation type by core orientation type
        Args:
            coreOrientationMode: core orientation type
        Returns:
            orientation type
        """
        orientationType = cls(coreOrientationMode.name)
        return orientationType

    def __repr__(self):
        return self.value


class OrientationMode(BaseEstimation):
    """
    Orientation mode. Estimates the image orientation.
    """

    #  pylint: disable=W0235
    def __init__(self, coreOrientationMode: CoreOrientationType):
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
            {"orientation_mode": self.orientationMode}
        """
        return {"orientation_mode": self.orientationMode}

    def getOrientationType(self) -> OrientationType:
        """
        Get orientation type
        Returns:
            orientation type
        """
        return OrientationType.fromCoreOrientationType(self._coreEstimation)


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
    def estimate(self, coreImage: CoreImage) -> OrientationMode:
        """
        Estimate orientation mode from warped image.

        Args:
            coreImage: core image

        Returns:
            estimated orientation mode
        Raises:
            LunaSDKException: if estimation is failed
        """
        error, orientationModeEstimation = self._coreEstimator.estimate(coreImage)

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

        return OrientationMode(orientationModeEstimation)
