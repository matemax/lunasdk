"""Module contains a face natural light estimator

See face_natural_light_.
"""
from typing import Union, List

from FaceEngine import (
    INaturalLightEstimatorPtr,
    LightStatus as LightStatusCore,
    NaturalLightEstimation as NaturalLightEstimationCore,
)  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...async_task import AsyncTask, DefaultPostprocessingFactory


class FaceNaturalLight(BaseEstimation):
    """
    Container for storing a estimated face natural light.

    Estimation properties:

        - status
        - score

    """

    #  pylint: disable=W0235
    def __init__(self, coreNaturalLight: NaturalLightEstimationCore):
        """
        Init.

        Args:
            coreNaturalLight:  face natural light estimation from core
        """
        super().__init__(coreNaturalLight)

    @property
    def status(self) -> bool:
        """
        Get a face natural light status.

        Returns:
            True if face has a natural light otherwise False

        """
        return True if self._coreEstimation.status == LightStatusCore.Natural else False

    @property
    def score(self) -> float:
        """Prediction score"""
        return self._coreEstimation.score

    def asDict(self):
        """
        Convert estimation to dict.

        Returns:
            dict with keys 'status' and 'score'
        """
        return {
            "status": self.status,
            "score": self.score,
        }


POST_PROCESSING = DefaultPostprocessingFactory(FaceNaturalLight)


class FaceNaturalLightEstimator(BaseEstimator):
    """
    Face natural light estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: INaturalLightEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    def estimate(
        self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: bool = False
    ) -> Union[FaceNaturalLight, AsyncTask[FaceNaturalLight]]:
        """
        Estimate natural face light on warp.

        Args:
            warp: warped image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated natural face light if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(warp.warpedImage.coreImage)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, estimation = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        return POST_PROCESSING.postProcessing(error, estimation)

    #  pylint: disable=W0221
    def estimateBatch(
        self, warps: List[Union[FaceWarp, FaceWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[FaceNaturalLight], AsyncTask[List[FaceNaturalLight]]]:
        """
        Estimate batch of face natural light.

        Args:
            warps:warped images
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated face natural light prediction if asyncEstimate is false otherwise async task
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
