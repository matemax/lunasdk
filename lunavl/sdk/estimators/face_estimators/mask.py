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
        Mask exists on the face and worn properly

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.maskInPlace

    @property
    def maskNotInPlace(self) -> float:
        """
        The mask exists on the face but is not worn properly

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.maskNotInPlace

    @property
    def noMask(self) -> float:
        """
        No mask on the face

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.noMask

    @property
    def occludedFace(self) -> float:
        """
        Face is occluded by other object (not by mask)

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.occludedFace

    def asDict(self) -> Dict[str, float]:
        """
        Convert to dict.

        Returns:
            {
                "mask_in_place": self.maskInPlace,
                "mask_not_in_place": self.maskNotInPlace,
                "no_mask": self.noMask,
                "occluded_face": self.occludedFace,
            }
        """
        return {
            "mask_in_place": self.maskInPlace,
            "mask_not_in_place": self.maskNotInPlace,
            "no_mask": self.noMask,
            "occluded_face": self.occludedFace,
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
