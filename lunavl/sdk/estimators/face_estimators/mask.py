"""Module for estimate a warped image quality.

See `warp quality`_.
"""
from enum import Enum
from typing import Union, Dict

from FaceEngine import MedicalMaskEstimation, IMedicalMaskEstimatorPtr  # pylint: disable=E0611,E0401
from FaceEngine import MedicalMask as CoreMask  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException
from ..base import BaseEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...base import BaseEstimation


class MaskState(Enum):
    """
    Emotions enum
    """

    #: Missing
    Missing = 1
    #: MedicalMask
    MedicalMask = 2
    #: Occluded
    Occluded = 3

    @staticmethod
    def fromCoreEmotion(coreMask: CoreMask) -> "MaskState":
        """
        Get enum element by core emotion.

        Args:
            coreEmotion: enum value from core

        Returns:
            corresponding emotion
        """
        if coreMask == CoreMask.NoMask:
            return MaskState.Missing
        if coreMask == CoreMask.Mask:
            return MaskState.MedicalMask
        if coreMask == CoreMask.OccludedFace:
            return MaskState.Occluded
        raise RuntimeError(f"bad core mask state {coreMask}")


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
    def medicalMask(self) -> float:
        """
        The probability that the mask exists on the face and is worn properly

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.maskScore

    @property
    def missing(self) -> float:
        """
        The probability that the mask not exists on the face

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.noMaskScore

    @property
    def occluded(self) -> float:
        """
        The probability that the face is occluded by other object (not by mask)

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.occludedFaceScore

    @property
    def predominateMask(self) -> MaskState:
        """
        Get predominate mask state.

        Returns:
            emotion with max score value
        """
        return MaskState.fromCoreEmotion(self._coreEstimation.result)

    def asDict(self) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Convert to dict.

        Returns:
            {
            "predominant_emotion": predominantName,
            "estimations": {
                "medical_mask": self.medicalMask,
                "missing": self.missing,
                "occluded": self.occluded,
            }
        }

        """
        predominant = self.predominateMask
        if predominant == MaskState.Occluded:
            predominantName = "occluded"
        elif predominant == MaskState.MedicalMask:
            predominantName = "medical_mask"
        else:
            predominantName = "missing"
        return {
            "predominant_mask": predominantName,
            "estimations": {"medical_mask": self.medicalMask, "missing": self.missing, "occluded": self.occluded,},
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
