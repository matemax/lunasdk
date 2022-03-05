"""Module contains a mouth state estimator

see `mouth state`_
"""
from typing import Union, Dict, List

from FaceEngine import MouthEstimation, IMouthEstimatorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, assertError
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...async_task import AsyncTask


class MouthStates(BaseEstimation):
    """
    Mouth states. There are 3 states of mouth: smile, occlusion and neither a smile nor an occlusion was detected.

    Estimation properties:

        - smile
        - mouth
        - occlusion
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: MouthEstimation):
        super().__init__(coreEstimation)

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
        """
        Convert to dict.

        Returns:
            {'opened': self.opened, 'occlusion': self.occlusion, 'smile': self.smile}
        """
        return {"opened": self.opened, "occluded": self.occlusion, "smile": self.smile}


def postProcessing(error, mouthState):
    assertError(error)
    return MouthStates(mouthState)


def postProcessingBatch(error, mouthStates):
    assertError(error)

    return [MouthStates(mouthState) for mouthState in mouthStates]


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
    @CoreExceptionWrap(LunaVLError.EstimationMouthStateError)
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
            task = self._coreEstimator.asyncEstimate(warp.warpedImage.coreImage)
            return AsyncTask(task, postProcessing)
        error, mouthState = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        return postProcessing(error, mouthState)

    #  pylint: disable=W0221
    @CoreExceptionWrap(LunaVLError.EstimationMouthStateError)
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
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, postProcessingBatch)
        error, masks = self._coreEstimator.estimate(coreImages)
        return postProcessingBatch(error, masks)
