from lunavl.sdk.estimators.base import BaseEstimator, ImageWithFaceDetection
from lunavl.sdk.async_task import AsyncTask
from FaceEngine import FSDKErrorResult, DynamicRangeEstimation
from lunavl.sdk.errors.exceptions import assertError
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator
from typing import List


def postProcessingBatch(error: FSDKErrorResult, dynamicRangeEstimations: List[DynamicRangeEstimation]) -> List[float]:
    """
    Post processing batch dynamic range estimation

    Args:
        error: estimation error
        dynamicRangeEstimations: list of dynamic range estimations

    Returns:
        list of dynamic range
    """
    assertError(error)
    return [estimation.dynamicRangeScore for estimation in dynamicRangeEstimations]


def postProcessing(error: FSDKErrorResult, dynamicRangeEstimator: DynamicRangeEstimation) -> float:
    """
    Post processing single dynamic range estimation

    Args:
        error: estimation error
        dynamicRangeEstimator: dynamic range estimation

    Returns:
        dynamic range
    """
    assertError(error)
    return dynamicRangeEstimator[0].dynamicRangeScore


class DynamicRangeEstimator(BaseEstimator):

    def estimate(
            self,
            imageWithFaceDetection: ImageWithFaceDetection,
            asyncEstimate: bool = False,
    ):
        """
        Estimate dynamic range from single image

        Args:
            imageWithFaceDetection: image with face detection
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated dynamic range or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(
                [imageWithFaceDetection.image.coreImage],
                [imageWithFaceDetection.boundingBox.coreEstimation]
            )
            return AsyncTask(task, postProcessing)
        error, dynamicRangeEstimation = self._coreEstimator.estimate(
            [imageWithFaceDetection.image.coreImage],
            [imageWithFaceDetection.boundingBox.coreEstimation]
        )
        return postProcessing(error, dynamicRangeEstimation)

    def estimateBatch(
            self,
            batch: ImageWithFaceDetection,
            asyncEstimate: bool = False,
    ):
        """
        Estimate dynamic range from single image

        Args:
            batch: list of image with face detection or face detections
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated dynamic range or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        coreImages = [row.image.coreImage for row in batch]
        detections = [row.boundingBox.coreEstimation for row in batch]
        validateInputByBatchEstimator(self._coreEstimator, coreImages, detections)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages, detections)
            return AsyncTask(task, postProcessingBatch)
        error, coreDynamicRangeList = self._coreEstimator.estimate(coreImages, detections)
        return postProcessingBatch(error, coreDynamicRangeList)
