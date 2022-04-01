"""Module contains a mouth state estimator

see `mouth state`_
"""
from enum import Enum
from typing import Union, Dict, List

from FaceEngine import MouthEstimation, IMouthEstimatorPtr, SmileType as CoreSmileType  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...async_task import AsyncTask, DefaultPostprocessingFactory


class SmileTypeEnum(Enum):
    """Smile type enum"""

    No = 0  # no smile
    Regular = 1  # regular smile, without teeths exposed (polite smile)
    WithTeeth = 2  # smile with teeths exposed

    @staticmethod
    def fromCoreSmile(coreSmile) -> "SmileTypeEnum":
        """
        Get enum element by core smile type.

        Args:
            coreSmile: enum value from core

        Returns:
            corresponding mask prediction
        """
        if coreSmile == getattr(CoreSmileType, "None"):
            return SmileTypeEnum.No
        if coreSmile == CoreSmileType.SmileLips:
            return SmileTypeEnum.Regular
        if coreSmile == CoreSmileType.SmileOpen:
            return SmileTypeEnum.WithTeeth
        raise RuntimeError(f"bad core smile type {coreSmile}")


class SmileType:
    """
    Smile type container
    Attributes:
        _coreSmileType: core smile type estimation
        _coreSmileScores:  smile type scores
    """

    __slots__ = ("_coreSmileType", "_coreSmileScores")

    def __init__(self, coreSmileType, coreSmileScores):
        self._coreSmileType = coreSmileType
        self._coreSmileScores = coreSmileScores

    @property
    def predominantType(self) -> SmileTypeEnum:
        """Get predominant smile type (if smile not found return SmileTypeEnum.No)"""
        return SmileTypeEnum.fromCoreSmile(self._coreSmileType)

    @property
    def regular(self):
        """Regular (polite) smile score"""
        return self._coreSmileScores.smileLips

    @property
    def withTeeth(self):
        """With teeth smile score"""
        return self._coreSmileScores.smileOpen

    def asDict(self) -> Dict:
        """Convert to dict"""
        predominant = "none" if self.predominantType == SmileTypeEnum.No else self.predominantType.name.lower()
        return {
            "estimations": {"regular": self.regular, "with_teeth": self.withTeeth},
            "predominant_type": predominant,
        }


class MouthProperties:
    """
    Container for mouth properties
    Attributes:
        smileType: smile type estimation
    """

    __slots__ = ("smileType",)

    def __init__(self, coreSmileType, coreSmileScores):
        self.smileType = SmileType(coreSmileType, coreSmileScores)

    def asDict(self) -> Dict:
        """Convert  mouth properties to dict"""
        return {"smile_type": self.smileType.asDict()}


class MouthStates(BaseEstimation):
    """
    Mouth states. There are 3 states of mouth: smile, occlusion and neither a smile nor an occlusion was detected.

    Estimation properties:

        - smile
        - mouth
        - occlusion
    """

    __slots__ = "properties"

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: MouthEstimation):
        super().__init__(coreEstimation)
        self.properties = MouthProperties(self._coreEstimation.smileType, self._coreEstimation.smileTypeScores)

    @property
    def smile(self) -> float:
        """
        Get smile score value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.smile

    @property
    def opened(self) -> float:
        """
        Get opened score value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.opened

    @property
    def occlusion(self) -> float:
        """
        Get occlusion score value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.occluded

    def asDict(self) -> Dict[str, float]:
        """ Convert to dict."""
        return {
            "opened": self.opened,
            "occluded": self.occlusion,
            "smile": self.smile,
            "properties": self.properties.asDict(),
        }


POST_PROCESSING = DefaultPostprocessingFactory(MouthStates)


class MouthStateEstimator(BaseEstimator):
    """
    Mouth state estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: IMouthEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    def estimate(
        self,
        warp: Union[FaceWarp, FaceWarpedImage],
        asyncEstimate: bool = False,
    ) -> Union[MouthStates, AsyncTask[MouthStates]]:
        """
        Estimate mouth state on warp.

        Args:
            warp: warped image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated states if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if asyncEstimate:
            task = self._coreEstimator.async_estimate_extended(warp.warpedImage.coreImage)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, mouthState = self._coreEstimator.estimate_extended(warp.warpedImage.coreImage)
        return POST_PROCESSING.postProcessing(error, mouthState)

    #  pylint: disable=W0221
    def estimateBatch(
        self, warps: List[Union[FaceWarp, FaceWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[MouthStates], AsyncTask[List[MouthStates]]]:
        """
        Batch estimate mouth states

        Args:
            warps: warped images
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated mouth states if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        coreImages = [warp.warpedImage.coreImage for warp in warps]

        validateInputByBatchEstimator(self._coreEstimator, coreImages)
        if asyncEstimate:
            task = self._coreEstimator.async_estimate_extended(coreImages)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        error, masks = self._coreEstimator.estimate_extended(coreImages)
        return POST_PROCESSING.postProcessingBatch(error, masks)
