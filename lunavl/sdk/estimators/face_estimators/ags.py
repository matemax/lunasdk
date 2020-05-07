"""Module contains an approximate garbage score estimator

See ags_.
"""
from typing import Optional

from FaceEngine import IAGSEstimatorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException
from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.image_utils.image import VLImage

from ..base import BaseEstimator
from ...base import BoundingBox


class AGSEstimator(BaseEstimator):
    """
    Approximate garbage score estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: IAGSEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    @CoreExceptionWrap(LunaVLError.EstimationAGSError)
    def estimate(
        self,
        detection: Optional[FaceDetection] = None,
        image: Optional[VLImage] = None,
        boundingBox: Optional[BoundingBox] = None,
    ) -> float:
        """
        Estimate emotion on warp.

        Args:
            image: image in R8G8B8 format
            boundingBox: face bounding box of corresponding the image
            detection: face detection

        Returns:
            estimated ags, float in range[0,1]
        Raises:
            LunaSDKException: if estimation failed
            ValueError: if image and detection is Noee
        """
        if detection is None:
            if image is None or boundingBox is None:
                raise ValueError("image and boundingBox or detection bust be not None")
            error, ags = self._coreEstimator.estimate(image.coreImage, boundingBox.coreEstimation)
        else:
            error, ags = self._coreEstimator.estimate(detection.image.coreImage, detection.boundingBox.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return ags
