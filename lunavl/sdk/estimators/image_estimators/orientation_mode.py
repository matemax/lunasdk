"""
Module contains an orientation mode estimator.

See `orientation mode`_.
"""
from enum import Enum
from typing import Union, List

from FaceEngine import IOrientationEstimatorPtr, OrientationType as CoreOrientationType, FSDKErrorResult

from lunavl.sdk.async_task import AsyncTask
from lunavl.sdk.errors.exceptions import assertError
from lunavl.sdk.estimators.base import BaseEstimator
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputByBatchEstimator
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarp, FaceWarpedImage
from lunavl.sdk.image_utils.image import VLImage


class OrientationType(Enum):
    """
    Enum for orientation type
    """

    #: normal-rotated image
    NORMAL = "Normal"
    #: left-rotated image | counter-clockwise
    LEFT = "Left"
    #: right-rotated image | clockwise
    RIGHT = "Right"
    #: upside-down image
    UPSIDE_DOWN = "UpsideDown"

    @classmethod
    def fromCoreOrientationType(cls, coreOrientationMode: CoreOrientationType) -> "OrientationType":
        """
        Create orientation type by core orientation type
        Args:
            coreOrientationMode: core orientation type
        Returns:
            orientation type
        """
        orientationType = cls(coreOrientationMode.name)
        return orientationType

    def __repr__(self):
        return self.value


def postProcessingBatch(error: FSDKErrorResult, orientations):
    """
    Post processing batch image orientation estimation
    Args:
        error:  estimation error
        orientations: estimated orientations

    Returns:
        list of `OrientationType`
    """
    assertError(error)

    return [OrientationType.fromCoreOrientationType(coreOrientationType) for coreOrientationType in orientations]


def postProcessing(error: FSDKErrorResult, orientationType):
    """
    Postprocessing single core image orientation estimation
    Args:
        error: estimation error
        orientationType: core estimation

    Returns:
        image orientation
    """
    assertError(error)

    return OrientationType.fromCoreOrientationType(orientationType)


class OrientationModeEstimator(BaseEstimator):
    """
    OrientationModeEstimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreOrientationModeEstimator: IOrientationEstimatorPtr):
        """
        Init.

        Args:
            coreOrientationModeEstimator: core orientation mode estimator
        """
        super().__init__(coreOrientationModeEstimator)

    def estimate(
        self, image: Union[VLImage, FaceWarp, FaceWarpedImage], asyncEstimate: bool = False
    ) -> Union[OrientationType, AsyncTask[OrientationType]]:
        """
        Estimate orientation mode from warped image.

        Args:
            image: vl image or face warp
             asyncEstimate: estimate or run estimation in background

        Returns:
            estimated orientation mode
        Raises:
            LunaSDKException: if estimation is failed
        """
        if isinstance(image, FaceWarp):
            coreImage = image.warpedImage.coreImage
        else:
            coreImage = image.coreImage
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImage)
            return AsyncTask(task, postProcessing)
        error, coreOrientationType = self._coreEstimator.estimate(coreImage)
        return postProcessing(error, coreOrientationType)

    def estimateBatch(
        self, images: List[Union[VLImage, FaceWarp, FaceWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[OrientationType], AsyncTask[List[OrientationType]]]:
        """
        Batch estimate orientation mode from warped images.

        Args:
            images: vl image or face warp list
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated orientation mode list if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation is failed
        """
        coreImages = [img.warpedImage.coreImage if isinstance(img, FaceWarp) else img.coreImage for img in images]

        validateInputByBatchEstimator(self._coreEstimator, coreImages)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, postProcessingBatch)
        error, coreOrientationTypeList = self._coreEstimator.estimate(coreImages)
        return postProcessingBatch(error, coreOrientationTypeList)
