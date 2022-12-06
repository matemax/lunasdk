from lunavl.sdk.estimators.base import BaseEstimator
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.async_task import AsyncTask
from FaceEngine import FSDKErrorResult, CrowdEstimation
from lunavl.sdk.errors.exceptions import assertError
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator
from typing import List


def postProcessingBatch(error: FSDKErrorResult, crowdEstimations: List[CrowdEstimation]) -> List[int]:
    """
    Post processing batch people count estimation

    Args:
        error: estimation error
        crowdEstimations: list of people count estimations

    Returns:
        list of people quantities
    """
    assertError(error)
    return [estimation.count for estimation in crowdEstimations]


def postProcessing(error: FSDKErrorResult, crowdEstimation: CrowdEstimation) -> int:
    """
    Post processing single people count estimation

    Args:
        error: estimation error
        crowdEstimations: people count estimation

    Returns:
        people count
    """
    assertError(error)
    return crowdEstimation[0].count


class PeopleCountEstimator(BaseEstimator):

    def estimate(self, image: VLImage, asyncEstimate: bool = False):
        """
        Estimate people count from single image

        Args:
            image: vl image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated people count or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate([image.coreImage], [image.coreImage.getRect()])
            return AsyncTask(task, postProcessing)
        error, crowdEstimation = self._coreEstimator.estimate([image.coreImage], [image.coreImage.getRect()])
        return postProcessing(error, crowdEstimation)

    def estimateBatch(self, images: List[VLImage], asyncEstimate: bool = False):
        """
        Estimate people count from single image

        Args:
            images: list of vl image
            asyncEstimate: estimate or run estimation in background

        Returns:
            list of estimated people count or async task if asyncEstimate is true
        Raises:
            LunaSDKException: if estimation is failed
        """
        coreImages = [image.coreImage for image in images]
        imgRects = [image.getRect() for image in coreImages]
        validateInputByBatchEstimator(self._coreEstimator, coreImages, imgRects)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, postProcessingBatch)
        error, crowdEstimations = self._coreEstimator.estimate(coreImages, imgRects)
        return postProcessingBatch(error, crowdEstimations)
