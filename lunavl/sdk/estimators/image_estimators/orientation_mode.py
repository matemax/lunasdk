"""
Module contains an orientation mode estimator.

See `orientation mode`_.
"""
from typing import Union

from FaceEngine import IOrientationEstimatorPtr, OrientationType

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, CoreExceptionWrap
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarp, FaceWarpedImage
from lunavl.sdk.estimators.base import BaseEstimator
from lunavl.sdk.image_utils.image import VLImage


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
    def estimate(self, image: Union[VLImage, FaceWarp, FaceWarpedImage]) -> OrientationType:
        """
        Estimate orientation mode from warped image.

        Args:
            image: vl image or face warp

        Returns:
            estimated orientation mode
        Raises:
            LunaSDKException: if estimation is failed
        """
        if isinstance(image, FaceWarp):
            coreImage = image.warpedImage.coreImage
        else:
            coreImage = image.coreImage

        error, orientationModeEstimation = self._coreEstimator.estimate(coreImage)

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

        return orientationModeEstimation
