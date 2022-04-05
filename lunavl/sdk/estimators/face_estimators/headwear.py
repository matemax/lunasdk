"""Module contains an headwear estimator

See headwear_.
"""
from enum import Enum
from typing import Union, List

from FaceEngine import (
    IHeadWearEstimatorPtr,
    HeadWearState as HeadWearStateCore,
    HeadWearTypeEstimation as HeadWearTypeCore,
    HeadWearEstimation as HeadWearCore,
)  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...async_task import AsyncTask, DefaultPostprocessingFactory


class HeadwearType(Enum):
    """
    A headgear type enum
    """

    #: no no headgear
    NoHeadWear = 0
    #: Baseball cap
    BaseballCap = 1
    #: Beanie
    Beanie = 2
    #: Peaked cap
    PeakedCap = 3
    #: Shawl
    Shawl = 4
    #: Hat with ear flaps
    HatWithEarFlaps = 5
    #: helmet
    Helmet = 6
    #: Hood
    Hood = 7
    # Hat
    Hat = 8
    # Other
    Other = 9

    @staticmethod
    def fromCoreHeadwear(coreHeadwearType: HeadWearTypeCore) -> "HeadwearType":
        """
        Get enum element by core headwear type.

        Args:
            coreHeadwearType: core  headwear type

        Returns:
            corresponding headwear type
        """
        return getattr(HeadwearType, coreHeadwearType.name)


class Headwear(BaseEstimation):
    """
    Container for storing a estimated headwear. List of headwear type expressions is represented in enum
    HeadwearType.

    Estimation properties:

        - type

    """

    #  pylint: disable=W0235
    def __init__(self, coreHeadwear: HeadWearCore):
        """
        Init.

        Args:
            coreHeadwear:  headwear estimation from core
        """
        super().__init__(coreHeadwear)

    @property
    def type(self) -> HeadwearType:
        return HeadwearType.fromCoreHeadwear(self._coreEstimation.type.result)

    def isWear(self) -> bool:
        return True if self._coreEstimation.state.result == HeadWearStateCore.Yes else False

    def asDict(self):
        """
        Convert estimation to dict.

        Returns:
            dict with keys 'type' and 'is_wear'
        """
        wearType = self.type
        if wearType == HeadwearType.NoHeadWear:
            wearTypeName = "none"
        elif wearType == HeadwearType.Hat:
            wearTypeName = "hat"
        elif wearType == HeadwearType.BaseballCap:
            wearTypeName = "baseball_cap"
        elif wearType == HeadwearType.Beanie:
            wearTypeName = "beanie"
        elif wearType == HeadwearType.HatWithEarFlaps:
            wearTypeName = "hat_with_ear_flaps"
        elif wearType == HeadwearType.Helmet:
            wearTypeName = "helmet"
        elif wearType == HeadwearType.Hood:
            wearTypeName = "hood"
        elif wearType == HeadwearType.PeakedCap:
            wearTypeName = "peaked_cap"
        elif wearType == HeadwearType.Shawl:
            wearTypeName = "shawl"
        else:
            wearTypeName = "other"
        return {
            "type": wearTypeName,
            "is_wear": self.isWear(),
        }


POST_PROCESSING = DefaultPostprocessingFactory(Headwear)


class HeadwearEstimator(BaseEstimator):
    """
    Headwear estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: IHeadWearEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    def estimate(
        self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: bool = False
    ) -> Union[Headwear, AsyncTask[Headwear]]:
        """
        Estimate headwear on warp.

        Args:
            warp: warped image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated headwear if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(warp.warpedImage.coreImage)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, headwear = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        return POST_PROCESSING.postProcessing(error, headwear)

    #  pylint: disable=W0221
    def estimateBatch(
        self, warps: List[Union[FaceWarp, FaceWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[Headwear], AsyncTask[List[Headwear]]]:
        """
        Batch estimate headwear

        Args:
            warps:warped images
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated headwears if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        coreImages = [warp.warpedImage.coreImage for warp in warps]

        validateInputByBatchEstimator(self._coreEstimator, coreImages)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        error, estimations = self._coreEstimator.estimate(coreImages)
        return POST_PROCESSING.postProcessingBatch(error, estimations)
