"""Module contains an approximate garbage score estimator

See ags_.
"""
from typing import Optional, List

from FaceEngine import IAGSEstimatorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException
from lunavl.sdk.detectors.facedetector import FaceDetection
from ..base import BaseEstimator, ImageWithFaceDetection
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator


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
        self, detection: Optional[FaceDetection] = None, imageWithFaceDetection: Optional[ImageWithFaceDetection] = None
    ) -> float:
        """
        Estimate ags for single image/detection.

        Args:
            detection: face detection
            imageWithFaceDetection: image with face detection

        Returns:
            estimated ags, float in range[0,1]
        Raises:
            LunaSDKException: if estimation failed
            ValueError: if image and detection are None
        """
        if detection is None:
            if imageWithFaceDetection is None:
                raise ValueError("image and boundingBox or detection must be not None")
            error, ags = self._coreEstimator.estimate(imageWithFaceDetection.image, imageWithFaceDetection.bBox)
        else:
            error, ags = self._coreEstimator.estimate(detection.image.coreImage, detection.boundingBox.coreEstimation)

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return ags

    @CoreExceptionWrap(LunaVLError.EstimationAGSError)
    def estimateAgsBatchByDetections(self, detections: List[FaceDetection]) -> List[float]:
        """
        Estimate ags for list of detections.

        Args:
            detections: face detection list

        Returns:
            list of estimated ags, float in range[0,1]
        Raises:
            LunaSDKException: if estimation failed
            ValueError: if empty image list and empty detection list or images count not match bounding boxes count
        """
        coreImages = [detection.image.coreImage for detection in detections]
        boundingBoxEstimations = [detection.boundingBox.coreEstimation for detection in detections]

        validateInputByBatchEstimator(self._coreEstimator, coreImages, boundingBoxEstimations)
        error, agsList = self._coreEstimator.estimate(coreImages, boundingBoxEstimations)

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return agsList

    @CoreExceptionWrap(LunaVLError.EstimationAGSError)
    def estimateAgsBatchByImages(self, imageWithFaceDetectionList: List[ImageWithFaceDetection]) -> List[float]:
        """
        Estimate ags for list of images with bounding boxes.

        Args:
            imageWithFaceDetectionList: list of image with face detection

        Returns:
            list of estimated ags, float in range[0,1]
        Raises:
            LunaSDKException: if estimation failed
            ValueError: if empty image list and empty detection list or images count not match bounding boxes count
        """
        argsMap = list(map(list, zip(*imageWithFaceDetectionList)))
        validateInputByBatchEstimator(self._coreEstimator, *argsMap)
        error, agsList = self._coreEstimator.estimate(*argsMap)

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return agsList
