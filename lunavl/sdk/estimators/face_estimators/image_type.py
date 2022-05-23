"""
Module contains an image color type estimator.

See `image color type`_.
"""
from enum import Enum
from typing import Dict, List, Union

from FaceEngine import (  # pylint: disable=E0611,E0401
    ImageColorEstimation as CoreImageColorEstimation,
    ImageColorType as CoreImageColorType,
)

from lunavl.sdk.base import BaseEstimation

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from .facewarper import FaceWarp, FaceWarpedImage


class ImageColorSchema(Enum):
    """
    A Image Color type enum
    """

    #: Grayscale
    Grayscale = 1
    #: Color
    Color = 2
    #: Infrared
    Infrared = 3

    @staticmethod
    def fromCoreEmotion(coreColorType: CoreImageColorType) -> "ImageColorSchema":
        """
        Get enum element by core image color type.

        Args:
            coreColorType: enum value from core

        Returns:
            corresponding image color type
        """
        return getattr(ImageColorSchema, coreColorType.name)


class ImageColorType(BaseEstimation):
    """
    Image color type container.

    Estimation properties:

        - grayscale
        - infrared
        - type
    """

    #  pylint: disable=W0235
    def __init__(self, coreImageColorType: CoreImageColorEstimation):
        """
        Init.

        Args:
            coreImageColorType: core fisheye estimation.
        """

        super().__init__(coreImageColorType)

    @property
    def grayscale(self) -> float:
        """Grayscale prediction score"""
        return self._coreEstimation.colorScore

    @property
    def infrared(self) -> float:
        """Infrared prediction score"""
        return self._coreEstimation.infraredScore

    @property
    def type(self) -> ImageColorSchema:
        """Prediction of image type."""
        return ImageColorSchema.fromCoreEmotion(self._coreEstimation.colorType)

    def asDict(self) -> Dict[str, Union[float, str]]:
        """Convert estimation to dict. """
        return {"grayscale": self.grayscale, "infrared": self.infrared, "type": self.type.name.lower()}


POST_PROCESSING = DefaultPostprocessingFactory(ImageColorType)


class ImageColorTypeEstimator(BaseEstimator):
    """
    Image color type estimator. Work on face detections. Allowed types see `ImageColorSchema`.
    """

    #  pylint: disable=W0221
    def estimate(  # type: ignore
        self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: bool = False
    ) -> Union[ImageColorType, AsyncTask[ImageColorType]]:
        """
        Estimate image color type on warp.

        Args:
            warp: warped image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated image color type if asyncEstimate is false otherwise async task
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
    ) -> Union[List[ImageColorType], AsyncTask[List[ImageColorType]]]:
        """
        Estimate image color type batch

        Args:
            warps:warped images
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated image color type if asyncEstimate is false otherwise async task
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
