"""Module for estimate a warped image quality.

See `warp quality`_.
"""
from typing import Dict, Union

# todo replace
from FaceEngine import SubjectiveQuality as CoreMask, IQualityEstimatorPtr  # pylint: disable=E0611,E0401
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException

from lunavl.sdk.base import BaseEstimation
from ..base import BaseEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage


class Mask(BaseEstimation):
    """
    Structure mask

    Estimation properties:

        - mask
    """

    #  pylint: disable=W0235
    def __init__(self, maskMock: CoreMask):
        """
        Init.

        Args:
            coreMask: estimated mask
        """
        super().__init__(maskMock)

    @property
    def mask(self) -> float:
        """
        Get mask.

        Returns:
            float in range(0, 1)
        """
        # todo replace
        return self._coreEstimation.blur

    def asDict(self) -> Dict[str, float]:
        """
        Convert to dict.

        Returns:
            {"mask": self.mask}
        """
        return {
            "mask": self.mask,
        }


class MaskEstimator(BaseEstimator):
    """
    Warp mask estimator.
    """

    #  pylint: disable=W0235
    # todo IQualityEstimatorPtr
    def __init__(self, coreEstimator: IQualityEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core mask estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    @CoreExceptionWrap(LunaVLError.EstimationMaskError)
    def estimate(self, warp: Union[FaceWarp, FaceWarpedImage]) -> Mask:
        """
        Estimate mask from a warp.

        Args:
            warp: raw warped image or warp

        Returns:
            estimated mask
        Raises:
            LunaSDKException: if estimation failed
        """
        error, mask = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return Mask(mask)
