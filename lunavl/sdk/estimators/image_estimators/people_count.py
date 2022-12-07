from lunavl.sdk.estimators.base import BaseEstimator
from lunavl.sdk.image_utils.image import VLImage, ColorFormat, CoreImage
from lunavl.sdk.image_utils.geometry import Rect, CoreRectI
from lunavl.sdk.async_task import AsyncTask
from lunavl.sdk.errors.exceptions import LunaSDKException, LunaVLError
from FaceEngine import FSDKErrorResult, CrowdEstimation
from lunavl.sdk.errors.exceptions import assertError
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator
from typing import List, Union, NamedTuple, Tuple

class ImageForPeopleEstimation(NamedTuple):
    """
    Structure for transfer image and detect area to people estimator

    Attributes:
        image: image for detection
        detectArea: area for people detection
    """

    image: VLImage
    detectArea: Rect


def assertImageFormat(image: VLImage):
    """
    Assert image for people estimation
    Args:
        image: image

    Raises:
        LunaSDKException: if image format is not R8G8B8
    """
    if image.format != ColorFormat.R8G8B8:
        details = f"Bad image format for people estimation, format: {image.format.value}, image: {image.filename}"
        raise LunaSDKException(LunaVLError.InvalidImageFormat.format(details))

def getEstimatorArgsFromImages(
        images: List[Union[VLImage, ImageForPeopleEstimation]]
) -> Tuple[List[CoreImage], List[CoreRectI]]:
    """
    Create args for people estimation from image list
    Args:
        images: list of images for estimation

    Returns:
        tuple: first - list core images
               second - detect area for corresponding images
    """

    coreImages, detectAreas = [], []

    for image in images:
        if isinstance(image, VLImage):
            img = image
            detectAreas.append(image.coreImage.getRect())
        else:
            img = image.image
            detectAreas.append(image.detectArea.coreRectI)
        # assertImageFormat(img)
        coreImages.append(img.coreImage)

    return coreImages, detectAreas


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

    def estimate(self, image: Union[VLImage, ImageForPeopleEstimation], asyncEstimate: bool = False):
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
        if isinstance(image, ImageForPeopleEstimation):
            detectArea = image.detectArea.coreRectI
            image = image.image
        else:
            detectArea = image.coreImage.getRect()
        assertImageFormat(image)
        validateInputByBatchEstimator(self._coreEstimator, [image.coreImage], [detectArea])
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate([image.coreImage], [detectArea])
            return AsyncTask(task, postProcessing)
        error, crowdEstimation = self._coreEstimator.estimate([image.coreImage], [detectArea])
        return postProcessing(error, crowdEstimation)

    def estimateBatch(self, images: List[Union[VLImage, ImageForPeopleEstimation]], asyncEstimate: bool = False):
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
        coreImages, detectAreas = getEstimatorArgsFromImages(images)
        validateInputByBatchEstimator(self._coreEstimator, coreImages, detectAreas)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, postProcessingBatch)
        error, crowdEstimations = self._coreEstimator.estimate(coreImages, detectAreas)
        return postProcessingBatch(error, crowdEstimations)
