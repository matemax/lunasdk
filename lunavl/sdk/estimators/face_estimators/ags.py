"""Module contains an approximate garbage score estimator

See ags_.
"""
from typing import Optional, List

from FaceEngine import IAGSEstimatorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException
from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.image_utils.image import VLImage
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputForBatchEstimator
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
        Estimate ags for single image/detection.

        Args:
            image: image in R8G8B8 format
            boundingBox: face bounding box of corresponding the image
            detection: face detection

        Returns:
            estimated ags, float in range[0,1]
        Raises:
            LunaSDKException: if estimation failed
            ValueError: if image and detection is None
        """
        if detection is None:
            if image is None or boundingBox is None:
                raise ValueError("image and boundingBox or detection must be not None")
            error, ags = self._coreEstimator.estimate(image.coreImage, boundingBox.coreEstimation)
        else:
            error, ags = self._coreEstimator.estimate(detection.image.coreImage, detection.boundingBox.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return ags

    @CoreExceptionWrap(LunaVLError.EstimationAGSError)
    def estimateAgsBatch(
        self,
        detections: Optional[List[FaceDetection]] = None,
        images: Optional[List[VLImage]] = None,
        boundingBoxes: Optional[List[BoundingBox]] = None,
    ) -> List[float]:
        """
        Estimate ags for list of images with bounding boxes/detections.

        Args:
            images: list of image in R8G8B8 format
            boundingBoxes: list of face bounding box of corresponding the images
            detections: list of face detection

        Returns:
            list of estimated ags, float in range[0,1]
        Raises:
            LunaSDKException: if estimation failed
            ValueError: if empty image list and empty detection list or images count not match bounding boxes count
        """
        if detections is None:
            if images is None:
                raise ValueError("required images or detections")
            if boundingBoxes is None:
                raise ValueError("required bounding boxes for images")
            coreImages = [image.coreImage for image in images]
            boundingBoxEstimations = [bbox.coreEstimation for bbox in boundingBoxes]
        else:
            coreImages = [detection.image.coreImage for detection in detections]
            boundingBoxEstimations = [detection.boundingBox.coreEstimation for detection in detections]

        validateInputForBatchEstimator(self._coreEstimator, coreImages, boundingBoxEstimations)
        error, agsList = self._coreEstimator.estimate(coreImages, boundingBoxEstimations)

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return agsList
