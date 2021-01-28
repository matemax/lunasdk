"""
Module contains an orientation mode estimator.

See `orientation mode`_.
"""

from FaceEngine import IOrientationEstimatorPtr, Image as CoreImage

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, CoreExceptionWrap
from ..base import BaseEstimator


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
    def estimate(self, coreImage: CoreImage) -> str:
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

        return orientationModeEstimation.name
