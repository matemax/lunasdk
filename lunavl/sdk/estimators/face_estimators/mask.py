"""Module for estimate a warped image quality.

See `warp quality`_.
"""
from enum import Enum
from typing import Dict, List, Literal, Union, overload

from FaceEngine import (  # pylint: disable=E0611,E0401; pylint: disable=E0611,E0401
    MedicalMask as CoreMask,
    MedicalMaskEstimation,
    DetailedMaskType,
)
from lunavl.sdk.detectors.facedetector import FaceDetection

from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ...base import BaseEstimation


class FaceOcclusionState(Enum):
    """
    Face occlusion by a mask enum
    """

    # no mask on the face
    Clear = "clear"
    # clear face with a mask around of a chin, mask does not cover anything in the face region (from mouth to eyes)
    Chin = "chin"
    # correct mask on the face (mouth and nose are covered correctly)
    Correct = "correct"
    # mask covers only a mouth
    Mouth = "mouth"
    # face is covered with not a medical mask or a full mask
    Partially = "partially"
    # face is covered with a full face mask (such as balaclava, sky mask, etc.)
    Full = "full"

    @staticmethod
    def fromCoreEmotion(coreFaceOcclusion: DetailedMaskType) -> "FaceOcclusionState":
        """
        Get enum element by core detailed mask type.

        Args:
            coreFaceOcclusion: enum value from core

        Returns:
            corresponding occlusion prediction
        """
        if coreFaceOcclusion == DetailedMaskType.CorrectMask:
            return FaceOcclusionState.Correct
        if coreFaceOcclusion == DetailedMaskType.MouthCoveredWithMask:
            return FaceOcclusionState.Mouth
        if coreFaceOcclusion == DetailedMaskType.ClearFace:
            return FaceOcclusionState.Clear
        if coreFaceOcclusion == DetailedMaskType.ClearFaceWithMaskUnderChin:
            return FaceOcclusionState.Chin
        if coreFaceOcclusion == DetailedMaskType.PartlyCoveredFace:
            return FaceOcclusionState.Partially
        if coreFaceOcclusion == DetailedMaskType.FullMask:
            return FaceOcclusionState.Full
        raise RuntimeError(f"bad core mask state {coreFaceOcclusion}")


class _FaceOcclusion(BaseEstimation):
    """
    Face occlusion by a mask estimation

    Estimation properties:

        - clear
        - chin
        - correct
        - mouth
        - partially
        - full

    Attributes:
        predominantOcclusion (FaceOcclusionState): predominant occlusion
    """

    #  pylint: disable=W0235
    def __init__(self, scores: list[float], predominantOcclusion: DetailedMaskType):
        """
        Init.
        Args:
            mask: estimated mask
        """
        self.predominantOcclusion = FaceOcclusionState.fromCoreEmotion(predominantOcclusion)
        super().__init__(scores)

    @property
    def clear(self) -> float:
        """No mask on the face score"""
        return self._coreEstimation[DetailedMaskType.ClearFace]

    @property
    def chin(self) -> float:
        """Clear face with a mask around of a chin score"""
        return self._coreEstimation[DetailedMaskType.ClearFaceWithMaskUnderChin]

    @property
    def correct(self) -> float:
        """Correct mask wearing score"""
        return self._coreEstimation[DetailedMaskType.CorrectMask]

    @property
    def mouth(self) -> float:
        """Mask covers only a mouth score"""
        return self._coreEstimation[DetailedMaskType.MouthCoveredWithMask]

    @property
    def partially(self) -> float:
        """Partialy face occlusion score. Face is occluded by not mask"""
        return self._coreEstimation[DetailedMaskType.PartlyCoveredFace]

    @property
    def full(self) -> float:
        """Full face mask score"""
        return self._coreEstimation[DetailedMaskType.FullMask]

    def asDict(self) -> dict:
        """Convert estimation to dict"""
        return {
            "predominant_occlusion": self.predominantOcclusion.value,
            "estimations": {
                "chin": self.chin,
                "mouth": self.mouth,
                "clear": self.clear,
                "correct": self.correct,
                "partially": self.partially,
                "full": self.full,
            },
        }


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
        Get enum element by core mask.

        Args:
            coreMask: enum value from core

        Returns:
            corresponding mask prediction
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
        self.faceOcclusion = _FaceOcclusion(mask.getScores(), mask.maskType)

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
            "predominant_mask": predominantName,
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
            "estimations": {"medical_mask": self.medicalMask, "missing": self.missing, "occluded": self.occluded},
            "face_occlusion": self.faceOcclusion.asDict(),
        }


POST_PROCESSING = DefaultPostprocessingFactory(Mask)


class MaskEstimator(BaseEstimator):
    """
    Warp mask estimator.
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self, faceObject: Union[FaceWarpedImage, FaceWarp, FaceDetection], asyncEstimate: Literal[False] = False
    ) -> Mask:  # type: ignore
        ...

    @overload
    def estimate(
        self, faceObject: Union[FaceWarpedImage, FaceWarp, FaceDetection], asyncEstimate: Literal[True]
    ) -> AsyncTask[Mask]:
        ...

    def estimate(
        self, faceObject: Union[FaceWarpedImage, FaceWarp, FaceDetection], asyncEstimate: bool = False
    ) -> Union[Mask, AsyncTask[Mask]]:
        """
        Estimate mask from a warp or detection.

        Args:
            faceObject: raw warped image, warp or faceDetection.
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated mask  if asyncEstimate is False otherwise async task

        Raises:
            LunaSDKException: if estimation failed
        """
        if isinstance(faceObject, (FaceWarpedImage, FaceWarp)):
            if asyncEstimate:
                task = self._coreEstimator.asyncEstimate(faceObject.warpedImage.coreImage)
                return AsyncTask(task, POST_PROCESSING.postProcessing)
            error, mask = self._coreEstimator.estimate(faceObject.warpedImage.coreImage)
        else:
            if asyncEstimate:
                task = self._coreEstimator.asyncEstimate(
                    faceObject.image.coreImage, faceObject.coreEstimation.detection
                )
                return AsyncTask(task, POST_PROCESSING.postProcessing)
            error, mask = self._coreEstimator.estimate(faceObject.image.coreImage, faceObject.coreEstimation.detection)
        return POST_PROCESSING.postProcessing(error, mask)

    #  pylint: disable=W0221
    def estimateBatch(
        self, warps: List[Union[FaceWarp, FaceWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[Mask], AsyncTask[List[Mask]]]:
        """
        Batch estimate mask from a warp.

        Args:
            warps: warp list
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated mask list if asyncEstimate is False otherwise async task

        Raises:
            LunaSDKException: if estimation failed
        """
        coreImages = [warp.warpedImage.coreImage for warp in warps]

        validateInputByBatchEstimator(self._coreEstimator, coreImages)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        error, masks = self._coreEstimator.estimate(coreImages)
        return POST_PROCESSING.postProcessingBatch(error, masks)
