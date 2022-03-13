"""Module contains an approximate garbage score estimator

See ags_.
"""
from typing import Optional, List, Union

from FaceEngine import IAGSEstimatorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.errors.exceptions import assertError
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
            error, ags = self._coreEstimator.estimate(
                imageWithFaceDetection.image.coreImage, imageWithFaceDetection.boundingBox.coreEstimation
            )
        else:
            error, ags = self._coreEstimator.estimate(detection.image.coreImage, detection.boundingBox.coreEstimation)

        assertError(error)
        return ags

    def estimateBatch(self, detections: Union[List[FaceDetection], List[ImageWithFaceDetection]]) -> List[float]:
        """
        Estimate ags for list of detections.

        Args:
            detections: face detection list or list of image with its face detection

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

        assertError(error)
        return agsList
