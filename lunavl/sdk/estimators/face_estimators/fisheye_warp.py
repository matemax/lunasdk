"""
Module contains a fisheye estimator.

See `fisheye`_.
"""
from typing import Dict, List, Union

from FaceEngine import FishEye as CoreFishEye, FishEyeEstimation as CoreFishEyeEstimation  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage


class Fisheye(BaseEstimation):
    """
    Fisheye. Estimation of fisheye effect on face detection (https://en.wikipedia.org/wiki/Fisheye_lens).

    Estimation properties:

        - status
        - score
    """

    #  pylint: disable=W0235
    def __init__(self, coreFishEye: CoreFishEyeEstimation):
        """
        Init.

        Args:
            coreFishEye: core fisheye estimation.
        """

        super().__init__(coreFishEye)

    @property
    def score(self) -> float:
        """Prediction score"""
        return self._coreEstimation.score

    @property
    def status(self) -> bool:
        """
        Prediction status.
        Returns:
            True if image contains the fisheye effect otherwise false
        """
        return True if self._coreEstimation.result == CoreFishEye.FishEyeEffect else False

    def asDict(self) -> Dict[str, Union[float, bool]]:
        """Convert estimation to dict."""
        return {"score": self.score, "status": self.status}


POST_PROCESSING = DefaultPostprocessingFactory(Fisheye)


class FisheyeEstimator(BaseEstimator):
    """
    Fisheye effect estimator. Work on face detections
    """

    def estimate(  # type: ignore
        self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: bool = False
    ) -> Union[Fisheye, AsyncTask[Fisheye]]:
        """
        Estimate fisheye.

        Args:
            warp: image with face detection
            asyncEstimate: estimate or run estimation in background
        Returns:
            fisheye estimation if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation is failed
        """
        if not asyncEstimate:
            error, estimation = self._coreEstimator.estimate_warp(warp.warpedImage.coreImage)
            return POST_PROCESSING.postProcessing(error, estimation)
        task = self._coreEstimator.asyncEstimate_warp(warp.warpedImage.coreImage)
        return AsyncTask(task, POST_PROCESSING.postProcessing)

    def estimateBatch(
        self, batch: List[Union[FaceWarp, FaceWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[Fisheye], AsyncTask[List[Fisheye]]]:
        """
        Estimate fisheye batch.

        Args:
            batch: list of image with face detection or face detections
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of fisheye estimations if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation is failed
        """

        coreImages = [row.warpedImage.coreImage for row in batch]

        validateInputByBatchEstimator(self._coreEstimator, coreImages)
        if not asyncEstimate:
            error, estimations = self._coreEstimator.estimate_warp(coreImages)
            return POST_PROCESSING.postProcessingBatch(error, estimations)
        task = self._coreEstimator.asyncEstimate_warp(coreImages)
        return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
