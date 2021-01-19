"""
Module contains a  livenessv1 estimator.

See `livenessv1`_.
"""
from enum import Enum
from typing import Optional, overload

from FaceEngine import LivenessOneShotRGBEstimation, ILivenessOneShotRGBEstimatorPtr  # pylint: disable=E0611,E0401
from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException
from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.estimators.base import BaseEstimator
from .head_pose import HeadPose


class LivenessPrediction(Enum):
    """
    Liveness estimation prediction
    """

    Real = "real"  # real human
    Spoof = "spoof"  # spoof
    Unknown = "unknown"  # unknown


class LivenessV1(BaseEstimation):
    """
    Liveness structure (LivenessOneShotRGBEstimation).

    Attributes:
        prediction: liveness prediction

    Estimation properties:

        - score
        - quality
    """

    __slots__ = ("prediction",)

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: LivenessOneShotRGBEstimation, prediction: LivenessPrediction):
        """
        Init.

        Args:
            coreEstimation: core estimation
        """
        self.prediction = prediction
        super().__init__(coreEstimation)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            {"prediction": self.prediction, "estimations": {"quality": self.quality, "score": self.score}}
        """
        return {"prediction": self.prediction.value, "estimations": {"quality": self.quality, "score": self.score}}

    @property
    def score(self) -> float:
        """
        Liveness score

        Returns:
            liveness score
        """
        return self._coreEstimation.score

    @property
    def quality(self) -> float:
        """
        Liveness quality score

        Returns:
            liveness quality score
        """
        return self._coreEstimation.qualityScore


class LivenessV1Estimator(BaseEstimator):
    """
    Liveness estimator version 1 (LivenessOneShotRGBEstimator).

    Attributes:
        principalAxes: maximum value of Yaw, pitch and roll angles for estimation
    """

    __slots__ = ("principalAxes",)

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: ILivenessOneShotRGBEstimatorPtr, principalAxes: float):
        """
        Init.

        Args:
            coreEstimator: core estimator
            principalAxes: maximum value of Yaw, pitch and roll angles for estimation
        """
        super().__init__(coreEstimator)
        self.principalAxes = principalAxes

    @overload  # type: ignore
    def estimate(self, faceDetection: FaceDetection) -> LivenessV1:
        """
        Estimate liveness by detection
        """
        ...

    @overload
    def estimate(self, faceDetection: FaceDetection, headPose: HeadPose) -> LivenessV1:  # type: ignore
        """
        Estimate liveness by detection with head pose validation
        """
        ...

    @overload
    def estimate(self, faceDetection: FaceDetection, yaw: float, pitch: float, roll) -> LivenessV1:  # type: ignore
        """
        Estimate liveness by detection with head pose validation
        """
        ...

    #  pylint: disable=W0221
    @CoreExceptionWrap(LunaVLError.EstimationEyesGazeError)
    def estimate(  # type: ignore
        self,
        faceDetection: FaceDetection,
        headPose: Optional[HeadPose] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
    ) -> LivenessV1:
        """
        Estimate a gaze direction

        .. warning::
            Current estimator version estimates correct liveness state for images from mobile and web camera only.
            A correctness of a liveness prediction are not guarantee for other images source.

        Args:
            faceDetection: face detection
            headPose: head pose
            yaw: yaw Tait–Bryan angle
            pitch: pitch Tait–Bryan angle
            roll: roll Tait–Bryan angle
        Returns:
            estimated liveness
        Raises:
            LunaSDKException: if estimation failed
        """
        if faceDetection.landmarks5 is None:
            raise ValueError("Landmarks5 is required for liveness estimation")
        error, estimation = self._coreEstimator.estimate(
            faceDetection.image.coreImage,
            faceDetection.coreEstimation.detection,
            faceDetection.landmarks5.coreEstimation,
        )
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        prediction = LivenessPrediction.Real if estimation.isReal else LivenessPrediction.Spoof
        if headPose:
            yaw = headPose.yaw
            roll = headPose.roll
            pitch = headPose.pitch
        if any(
            (
                yaw is not None and self.principalAxes < abs(yaw),
                pitch is not None and self.principalAxes < abs(pitch),
                roll is not None and self.principalAxes < abs(roll),
            )
        ):
            prediction = LivenessPrediction.Unknown

        return LivenessV1(estimation, prediction)
