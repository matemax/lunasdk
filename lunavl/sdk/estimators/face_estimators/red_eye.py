"""Module contains a red-eye estimator

See face_natural_light_.
"""
from typing import List, Union

from FaceEngine import (  # pylint: disable=E0611,E0401
    RedEyeAttributes as RedEyeCore,
    RedEyeEstimation as RedEyeEstimationCore,
    RedEyeStatus as RedEyeStatusCore,
)

from lunavl.sdk.base import BaseEstimation

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from .eyes import WarpWithLandmarks5


class RedEye(BaseEstimation):
    """
    Container for storing a estimated red-eye for one eye.

    Estimation properties:

        - status
        - score

    """

    #  pylint: disable=W0235
    def __init__(self, coreRedEye: RedEyeCore):
        """
        Init.

        Args:
            coreRedEye: red-eye estimation from core
        """
        super().__init__(coreRedEye)

    @property
    def status(self) -> bool:
        """
        Get a red-eye status.

        Returns:
            True if eye has red-eye effect otherwise False

        """
        return True if self._coreEstimation.status == RedEyeStatusCore.Red else False

    @property
    def score(self) -> float:
        """Prediction score"""
        return self._coreEstimation.score

    def asDict(self):
        """
        Convert estimation to dict.

        Returns:
            dict with keys 'status' and 'score'
        """
        return {
            "status": self.status,
            "score": self.score,
        }


class RedEyes(BaseEstimation):
    """
    Red-eyes estimation structure for both eyes.

    Attributes:
        leftEye (Eye): estimation for left eye
        rightEye (Eye): estimation for right eye
    """

    __slots__ = ("leftEye", "rightEye")

    def __init__(self, coreEstimation: RedEyeEstimationCore):
        """
        Init.

        Args:
            coreEstimation: core estimation
        """
        super().__init__(coreEstimation)
        self.leftEye = RedEye(coreEstimation.leftEye)
        self.rightEye = RedEye(coreEstimation.rightEye)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            {'left_eye': self.leftEye, 'right_eye': self.rightEye}
        """
        return {"left_eye": self.leftEye.asDict(), "right_eye": self.rightEye.asDict()}


POST_PROCESSING = DefaultPostprocessingFactory(RedEyes)


class RedEyesEstimator(BaseEstimator):
    """
    Red-eye estimator.
    """

    #  pylint: disable=W0221
    def estimate(  # type: ignore
        self, warpWithLandmarks5: WarpWithLandmarks5, asyncEstimate: bool = False
    ) -> Union[RedEyes, AsyncTask[RedEyes]]:
        """
        Estimate red-eye on warp.

        Args:
            warpWithLandmarks5: warped image with transformed landmarks
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated red-eye statuses if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        image = warpWithLandmarks5.warp.warpedImage.coreImage
        landmarks = warpWithLandmarks5.landmarks.coreEstimation
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(image, landmarks)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, gaze = self._coreEstimator.estimate(image, landmarks)
        return POST_PROCESSING.postProcessing(error, gaze)

    #  pylint: disable=W0221
    def estimateBatch(
        self, warpWithLandmarks5List: List[WarpWithLandmarks5], asyncEstimate: bool = False
    ) -> Union[List[RedEyes], AsyncTask[List[RedEyes]]]:
        """
        Estimate batch of red-eye.

        Args:
            warpWithLandmarks5List: warps with landmarks
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of red-eye prediction if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        images = [row.warp.warpedImage.coreImage for row in warpWithLandmarks5List]
        landmarks = [row.landmarks.coreEstimation for row in warpWithLandmarks5List]
        validateInputByBatchEstimator(self._coreEstimator, images, landmarks)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(images, landmarks)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        error, estimations = self._coreEstimator.estimate(images, landmarks)
        return POST_PROCESSING.postProcessingBatch(error, estimations)
