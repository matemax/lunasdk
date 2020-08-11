"""Module for estimate a warped image quality.

See `warp quality`_.
"""
from typing import Union, Dict

from FaceEngine import MedicalMaskEstimation, IMedicalMaskEstimatorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException
from ..base import BaseEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...base import BaseEstimation


class Mask(BaseEstimation):
    """
    Structure mask

    Estimation properties:

        - mask
    """

    #  pylint: disable=W0235
    def __init__(self, mask: MedicalMaskEstimation):
        """
        Init.

        Args:
            mask: estimated mask
        """
        super().__init__(mask)

    @property
    def maskInPlace(self) -> float:
        """
        Get "mask in place" probability.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.maskInPlace

    @property
    def isMaskInPlace(self) -> bool:
        """
        Get "mask is place".

        Returns:
            bool
        """
        return self._coreEstimation.isMaskInPlace

    @property
    def maskNotInPlace(self) -> float:
        """
        Get "mask not in place" probability.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.maskNotInPlace

    @property
    def isMaskNotInPlace(self) -> bool:
        """
        Get "mask not in place".

        Returns:
            bool
        """
        return self._coreEstimation.isMaskNotInPlace

    @property
    def noMask(self) -> float:
        """
        Get "no mask" probability.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.noMask

    @property
    def isNoMask(self) -> bool:
        """
        Get "no mask".

        Returns:
            bool
        """
        return self._coreEstimation.isNoMask

    @property
    def occludedFace(self) -> float:
        """
        Get "occluded face" probability.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.occludedFace

    @property
    def isOccludedFace(self) -> bool:
        """
        Get "occluded face".

        Returns:
            bool
        """
        return self._coreEstimation.isOccludedFace

    def asDict(self) -> Dict[str, float]:
        """
        Convert to dict.

        Returns:
            {"score": self.maskInPlace}
        """
        return {
            "mask_in_place": self.maskInPlace,
            "is_mask_in_place": self.isMaskInPlace,
            "mask_not_in_place": self.maskNotInPlace,
            "is_mask_not_in_place": self.isMaskNotInPlace,
            "no_mask": self.noMask,
            "is_no_mask": self.isNoMask,
            "occluded_face": self.occludedFace,
            "is_occluded_face": self.isOccludedFace,
        }


class MaskEstimator(BaseEstimator):
    """
    Warp mask estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, maskEstimator: IMedicalMaskEstimatorPtr):
        """
        Init.

        Args:
            maskEstimator: core mask estimator
        """
        super().__init__(maskEstimator)

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
