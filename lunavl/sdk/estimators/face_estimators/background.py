"""
Module contains a background estimator.

See `background`_.
"""
from typing import Dict, List, Union

from FaceEngine import (
    IBackgroundEstimatorPtr,
    BackroundEstimation as CoreBackgroundEstimation,
    BackgroundStatus as CoreBackgroundStatus,
)  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.detectors.facedetector import FaceDetection
from ..base import BaseEstimator, ImageWithFaceDetection
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ...async_task import AsyncTask, DefaultPostprocessingFactory


class FaceDetectionBackground(BaseEstimation):
    """
    Background. Estimation of face background on image.

    Estimation properties:

        - status
        - score
    """

    #  pylint: disable=W0235
    def __init__(self, coreBackground: CoreBackgroundEstimation):
        """
        Init.

        Args:
            coreBackground: core background estimation.
        """

        super().__init__(coreBackground)

    @property
    def backgroundColor(self) -> float:
        """Background color score"""
        return self._coreEstimation.backgroundColorScore

    @property
    def solidColor(self) -> float:
        """Background solid color score, 1 - is uniform background, 0 - is non uniform"""
        return self._coreEstimation.backgroundScore

    @property
    def lightBackground(self) -> float:
        """
        Background light score, 1 - is light background, 0 - is too dark.
        """
        return self._coreEstimation.backgroundColorScore

    @property
    def status(self) -> bool:
        """
        Prediction status.
        Returns:
            True if background is solid and light otherwise False
        """
        return True if self._coreEstimation.status == CoreBackgroundStatus.Solid else False

    def asDict(self) -> Dict[str, Union[float, bool]]:
        """Convert estimation to dict. """
        return {"light_background": self.lightBackground, "status": self.status, "solid_color": self.solidColor}


POST_PROCESSING = DefaultPostprocessingFactory(FaceDetectionBackground)


class FaceDetectionBackgroundEstimator(BaseEstimator):
    """
    Face detection background estimator. Work on face detections
    """

    #  pylint: disable=W0235
    def __init__(self, corBackgroundEstimator: IBackgroundEstimatorPtr):
        """
        Init.

        Args:
            corBackgroundEstimator: core estimator
        """
        super().__init__(corBackgroundEstimator)

    def estimate(
        self, imageWithFaceDetection: ImageWithFaceDetection, asyncEstimate: bool = False
    ) -> Union[FaceDetectionBackground, AsyncTask[FaceDetectionBackground]]:
        """
        Estimate a face detection background.

        Args:
            imageWithFaceDetection: image with face detection
            asyncEstimate: estimate or run estimation in background
        Returns:
            background estimation if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation is failed
        """
        if not asyncEstimate:
            error, estimation = self._coreEstimator.estimate(
                imageWithFaceDetection.image.coreImage, imageWithFaceDetection.boundingBox.coreEstimation
            )
            return POST_PROCESSING.postProcessing(error, estimation)
        task = self._coreEstimator.asyncEstimate(
            imageWithFaceDetection.image.coreImage, imageWithFaceDetection.boundingBox.coreEstimation
        )
        return AsyncTask(task, POST_PROCESSING.postProcessing)

    def estimateBatch(
        self, batch: Union[List[ImageWithFaceDetection], List[FaceDetection]], asyncEstimate: bool = False
    ) -> Union[List[FaceDetectionBackground], AsyncTask[List[FaceDetectionBackground]]]:
        """
        Estimate background batch.

        Args:
            batch: list of image with face detection or face detections
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of background estimations if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation is failed
        """

        coreImages = [row.image.coreImage for row in batch]
        detections = [row.boundingBox.coreEstimation for row in batch]

        validateInputByBatchEstimator(self._coreEstimator, coreImages, detections)
        if not asyncEstimate:
            error, estimations = self._coreEstimator.estimate(coreImages, detections)
            return POST_PROCESSING.postProcessingBatch(error, estimations)
        task = self._coreEstimator.asyncEstimate(coreImages, detections)
        return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
