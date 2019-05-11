"""
Module contains a mouth state estimator

See `eyes`_ and `gaze direction`_.

"""
from enum import Enum
from typing import Union

from FaceEngine import IEyeEstimatorPtr, EyeCropper, IGazeEstimatorPtr  # pylint: disable=E0611,E0401
from FaceEngine import EyelidLandmarks as CoreEyelidLandmarks  # pylint: disable=E0611,E0401
from FaceEngine import IrisLandmarks as CoreIrisLandmarks  # pylint: disable=E0611,E0401
from FaceEngine import State as CoreEyeState, EyesEstimation as CoreEyesEstimation  # pylint: disable=E0611,E0401
from FaceEngine import EyeAngles, GazeEstimation as CoreGazeEstimation   # pylint: disable=E0611,E0401

from lunavl.sdk.estimators.base_estimation import BaseEstimation, BaseEstimator
from lunavl.sdk.estimators.face_estimators.head_pose import HeadPose
from lunavl.sdk.faceengine.facedetector import Landmarks5, Landmarks68

from lunavl.sdk.estimators.face_estimators.warper import Warp, WarpedImage
from lunavl.sdk.image_utils.geometry import Landmarks


class EyeState(Enum):
    """
    Enum for eye states.
    """
    #: eye is opened
    Open = 1
    #: eye is occluded
    Occluded = 2
    #: eye is closed
    Closed = 3

    @staticmethod
    def fromCoreEmotion(coreEyeState: CoreEyeState) -> 'EyeState':
        """
        Get enum element by core emotion.

        Args:
            coreEyeState: an eye state form core

        Returns:
            corresponding eye state
        """
        return getattr(EyeState, coreEyeState.name)


class IrisLandmarks(Landmarks):
    """
     Eyelid landmarks.
    """

    #  pylint: disable=W0235
    def __init__(self, coreIrisLandmarks: CoreIrisLandmarks):
        """
        Init

        Args:
            coreIrisLandmarks: core iris landmarks
        """
        super().__init__(coreIrisLandmarks)


class EyelidLandmarks(Landmarks):
    """
     Eyelid landmarks.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEyelidLandmarks: CoreEyelidLandmarks):
        """
        Init

        Args:
            coreEyelidLandmarks: core  eyelid landmarks
        """
        super().__init__(coreEyelidLandmarks)


class Eye(BaseEstimation):
    """
    Eye structure.

    Estimation properties:

        - eyelid
        - iris
    """
    __slots__ = ("irisLandmarks", "eyelidLandMarks", "state")

    #  pylint: disable=W0235
    def __init__(self, coreEstimation):
        """
        Init.

        Args:
            coreEstimation: core estimation
        """
        super().__init__(coreEstimation)
        self.irisLandmarks = IrisLandmarks(self._coreEstimation.iris)
        self.eyelidLandMarks = EyelidLandmarks(self._coreEstimation.eyelid)
        self.state = EyeState.fromCoreEmotion(self._coreEstimation.state)

    @property
    def eyelid(self) -> EyelidLandmarks:
        """
        Get eyelid landmarks.

        Returns:
            eyelid landmarks
        """
        return self.eyelidLandMarks

    @property
    def iris(self) -> IrisLandmarks:
        """
        Get iris landmarks.

        Returns:
            iris landmarks
        """
        return self.irisLandmarks

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            {"iris_landmarks": self.irisLandmarks.asDict(), "eyelid_landmarks": self.eyelidLandMarks.asDict(),
             "state": self.state.name.lower()}
        """
        return {"iris_landmarks": self.irisLandmarks.asDict(),
                "eyelid_landmarks": self.eyelidLandMarks.asDict(),
                "state": self.state.name.lower()}


class EyesEstimation(BaseEstimation):
    """
    Eyes estimation structure.

    Attributes:
        leftEye (Eye): estimation for left eye
        rightEye (Eye): estimation for right eye
    """
    __slots__ = ("leftEye", "rightEye")

    def __init__(self, coreEstimation: CoreEyesEstimation):
        """
        Init.

        Args:
            coreEstimation: core estimation
        """
        super().__init__(coreEstimation)
        self.leftEye = Eye(coreEstimation.leftEye)
        self.rightEye = Eye(coreEstimation.rightEye)

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            {'yaw': self.leftEye, 'pitch': self.rightEye}
        """
        return {"left_eye": self.leftEye.asDict(),
                "right_eye": self.rightEye.asDict()}


class EyeEstimator(BaseEstimator):
    """
    Eye estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: IEyeEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    def estimate(self, transformedLandmarks: Union[Landmarks5, Landmarks68],
                 warp: Union[Warp, WarpedImage]) -> EyesEstimation:
        """
        Estimate mouth state on warp.

        Args:
            warp: warped image
            transformedLandmarks: transformed landmarks

        Returns:
            estimated states
        """
        cropper = EyeCropper()
        if isinstance(transformedLandmarks, Landmarks5):
            eyeRects = cropper.cropByLandmarks5(warp.warpedImage.coreImage,
                                                transformedLandmarks.coreEstimation)
        else:
            eyeRects = cropper.cropByLandmarks68(warp.warpedImage.coreImage,
                                                 transformedLandmarks.coreEstimation)
        error, eyesEstimation = self._coreEstimator.estimate(warp.warpedImage.coreImage, eyeRects)
        if error.isError:
            raise ValueError("12343")
        return EyesEstimation(eyesEstimation)


class GazeDirection(BaseEstimation):
    """
    Gaze direction structure.
    Estimation properties:

        - yaw
        - pitch
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: EyeAngles):
        """
        Init.
        """
        super().__init__(coreEstimation)

    @property
    def yaw(self) -> float:
        """
        Get the yaw angle.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.yaw

    @property
    def pitch(self) -> float:
        """
        Get the pitch angle.

        Returns:
            float in range(0, 1)
        """
        return self._coreEstimation.pitch

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            {'yaw': self.yaw, 'pitch': self.pitch}
        """
        return {'yaw': self.yaw, 'pitch': self.pitch}


class GazeEstimation(BaseEstimation):
    """
    Gaze estimation.

    Attributes:
        leftEye (GazeDirection): left eye gaze direction
        rightEye (GazeDirection): right eye gaze direction
    """
    __slots__ = ("leftEye", "rightEye")

    def __init__(self, coreEstimation: CoreGazeEstimation):
        """
        Init.

        Args:
            coreEstimation: core estimation
        """
        super().__init__(coreEstimation)
        self.leftEye = GazeDirection(coreEstimation.leftEye)
        self.rightEye = GazeDirection(coreEstimation.rightEye)

    def asDict(self) -> dict:
        """
        Convert self to a dict.

        Returns:
            {"left_eye": self.leftEye.asDict(), "right_eye": self.rightEye.asDict()}
        """
        return {"left_eye": self.leftEye.asDict(), "right_eye": self.rightEye.asDict()}


class GazeEstimator(BaseEstimator):
    """
    Gaze direction estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: IGazeEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    def estimate(self, headPose: HeadPose, eyesEstimation: EyesEstimation) -> GazeEstimation:
        """
        Estimate a gaze direction

        Args:
            headPose: head pose (calculated using landmarks68)
            eyesEstimation: eyes estimation
        Returns:
            estimated states
        """
        error, gaze = self._coreEstimator.estimate(headPose.coreEstimation, eyesEstimation.coreEstimation)
        if error.isError:
            raise ValueError("12343")
        return GazeEstimation(gaze)
