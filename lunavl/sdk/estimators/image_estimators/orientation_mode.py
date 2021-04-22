"""
Module contains an orientation mode estimator.

See `orientation mode`_.
"""
from enum import Enum
from typing import Union, List

from FaceEngine import IOrientationEstimatorPtr, OrientationType as CoreOrientationType

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, CoreExceptionWrap
from lunavl.sdk.estimators.estimators_utils.extractor_utils import validateInputForBatchEstimator
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarp, FaceWarpedImage
from lunavl.sdk.estimators.base import BaseEstimator
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

    @CoreExceptionWrap(LunaVLError.EstimationOrientationModeError)
    def estimate(self, image: Union[VLImage, FaceWarp, FaceWarpedImage]) -> OrientationType:
        """
        Estimate orientation mode from warped image.

        Args:
            image: vl image or face warp

        Returns:
            estimated orientation mode
        Raises:
            LunaSDKException: if estimation is failed
        """
        if isinstance(image, FaceWarp):
            coreImage = image.warpedImage.coreImage
        else:
            coreImage = image.coreImage

        error, coreOrientationType = self._coreEstimator.estimate(coreImage)

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

        return OrientationType.fromCoreOrientationType(coreOrientationType)

    @CoreExceptionWrap(LunaVLError.EstimationOrientationModeError)
    def estimateBatch(self, images: List[Union[VLImage, FaceWarp, FaceWarpedImage]]) -> List[OrientationType]:
        """
        Batch estimate orientation mode from warped images.

        Args:
            images: vl image or face warp list

        Returns:
            estimated orientation mode list
        Raises:
            LunaSDKException: if estimation is failed
        """
        coreImages = [img.warpedImage.coreImage if isinstance(img, FaceWarp) else img.coreImage for img in images]

        validateInputForBatchEstimator(self._coreEstimator, coreImages)
        error, coreOrientationTypeList = self._coreEstimator.estimate(coreImages)

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

        return [
            OrientationType.fromCoreOrientationType(coreOrientationType)
            for coreOrientationType in coreOrientationTypeList
        ]
