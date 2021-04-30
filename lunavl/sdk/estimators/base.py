"""
Module with base classes of estimators and estimations
"""
from abc import ABC, abstractmethod
from typing import Any, NamedTuple

from lunavl.sdk.base import BoundingBox
from lunavl.sdk.image_utils.image import VLImage


class BaseEstimator(ABC):
    """
    Base estimator class. Class is  a container for core estimations. Mostly estimate attributes  can be get through
    a corresponding properties.

    Attributes:
        _coreEstimator: core estimator
    """

    __slots__ = ("_coreEstimator",)

    def __init__(self, coreEstimator):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        self._coreEstimator = coreEstimator

    @abstractmethod
    def estimate(self, *args, **kwargs) -> Any:
        """
        Estimate attributes on warp.

        Returns:
            estimated attributes
        """
        pass


class ImageWithFaceDetection(NamedTuple):
    """
    Structure for transferring an image and its detection.
    Attributes
        image (VLImage): image for detection
        boundingBox (BoundingBox): face bounding box
    """

    image: VLImage
    boundingBox: BoundingBox
