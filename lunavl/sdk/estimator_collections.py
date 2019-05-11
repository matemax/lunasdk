"""Module contains estimator collections.
"""
from enum import Enum
from typing import List, Optional

from lunavl.sdk.estimators.face_estimators.basic_attributes import BasicAttributesEstimator
from lunavl.sdk.estimators.face_estimators.emotions import EmotionsEstimator
from lunavl.sdk.estimators.face_estimators.eyes import EyeEstimator, GazeEstimator
from lunavl.sdk.estimators.face_estimators.head_pose import HeadPoseEstimator
from lunavl.sdk.estimators.face_estimators.mouth_state import MouthStateEstimator
from lunavl.sdk.estimators.face_estimators.warp_quality import WarpQualityEstimator
from lunavl.sdk.faceengine.engine import VLFaceEngine, FACE_ENGINE


class FaceEstimator(Enum):
    """
    Enum for face estimators.
    """
    #: head pose estimator
    HeadPose = 1
    #: eye estimation (eyelid, iris landmarks and state)
    Eye = 2
    #: emotion estimator
    Emotions = 3
    #: basic attributes estimator
    BasicAttributes = 4
    #: gaze direction estimator
    GazeDirection = 5
    #: mouth state estimator
    MouthState = 6
    #: warp quality estimator
    WarpQuality = 7


class FaceEstimatorsCollection:
    """
    Collection of lazy load face estimators.

    Attributes:
        _headPoseEstimator (Optional[HeadPoseEstimator]): lazy load head pose estimator
        _eyeEstimator (Optional[EyeEstimator]): lazy load eye estimator
        _gazeDirectionEstimator (Optional[GazeEstimator]): lazy load gaze direction estimator
        _mouthStateEstimator (Optional[MouthStateEstimator]): lazy load mouth state estimator
        _warpQualityEstimator (Optional[WarpQualityEstimator]): lazy load warp quality estimator
        _basicAttributesEstimator (Optional[BasicAttributesEstimator]): lazy load basic attributes estimator
        _emotionsEstimator (Optional[EmotionsEstimator]): lazy load emotions estimator
        warper (Optional[Warper]): warper
    """
    __slots__ = ("_headPoseEstimator", "_eyeEstimator", "_gazeDirectionEstimator", "_mouthStateEstimator",
                 "_warpQualityEstimator", "_basicAttributesEstimator", "_emotionsEstimator", "_faceEngine",
                 "warper")

    def __init__(self, startEstimators: Optional[List[FaceEstimator]] = None,
                 faceEngine: Optional[VLFaceEngine] = None):
        """
        Init.

        Args:
            startEstimators: list of estimators which will be initiate now
            faceEngine: faceengine, factory for estimators
        """
        if faceEngine is None:
            self._faceEngine = FACE_ENGINE
        else:
            self._faceEngine = faceEngine

        self._basicAttributesEstimator = None
        self._eyeEstimator = None
        self._emotionsEstimator = None
        self._gazeDirectionEstimator = None
        self._mouthStateEstimator = None
        self._warpQualityEstimator = None
        self._headPoseEstimator = None
        self.warper = self._faceEngine.createWarper()

        if startEstimators:
            for estimator in set(startEstimators):
                self.initEstimator(estimator)

    @staticmethod
    def _getEstimatorByAttributeName(estimatorAttributeName: str) -> FaceEstimator:
        """
        Get face estimator by attribute name which corresponding to estimator
        Args:
            estimatorAttributeName: name

        Returns:
            corresponding face estimator
        Raises:
            ValueError("Bad attribute name"): if face estimator not found
        """
        for estimator in FaceEstimator:
            if estimator.name.lower() == estimatorAttributeName[1: -len("Estimator")]:
                return estimator
        raise ValueError("Bad attribute name")

    def _getAttributeNameByEstimator(self, estimator: FaceEstimator) -> str:
        """
        Get an estimator attribute name bya face estimator

        Args:
            estimator: face estimator

        Returns:
            corresponding attribute name
        Raises:
            ValueError("Bad estimator"): if attribute name not found

        """
        for estimatorName in self.__slots__:
            if estimatorName == "_faceEngine":
                continue
            if estimator.name.lower() == estimatorName[1: -len("Estimator")]:
                return estimatorName
        raise ValueError("Bad estimator")

    def initEstimator(self, estimator: FaceEstimator) -> None:
        """
        Create an estimator. Create new estimator with help self._faceengine

        Args:
            estimator: estimator for creating
        Raises:
             ValueError("Bad estimator type"): if estimator not found
        """
        if estimator == FaceEstimator.BasicAttributes:
            self._basicAttributesEstimator = self._faceEngine.createBasicAttributesEstimator()
        elif estimator == FaceEstimator.Eye:
            self._eyeEstimator = self._faceEngine.createEyeEstimator()
        elif estimator == FaceEstimator.Emotions:
            self._emotionsEstimator = self._faceEngine.createEmotionEstimator()
        elif estimator == FaceEstimator.GazeDirection:
            self._gazeDirectionEstimator = self._faceEngine.createGazeEstimator()
        elif estimator == FaceEstimator.MouthState:
            self._mouthStateEstimator = self._faceEngine.createMouthEstimator()
        elif estimator == FaceEstimator.WarpQuality:
            self._warpQualityEstimator = self._faceEngine.createWarpQualityEstimator()
        elif estimator == FaceEstimator.HeadPose:
            self._headPoseEstimator = self._faceEngine.createHeadPoseEstimator()
        else:
            raise ValueError("Bad estimator type")

    @property
    def headPoseEstimator(self) -> HeadPoseEstimator:
        """
        Get head pose estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            estimator
        """
        if self._headPoseEstimator is None:
            self._headPoseEstimator = self._faceEngine.createHeadPoseEstimator()
        return self._headPoseEstimator

    @headPoseEstimator.setter
    def headPoseEstimator(self, newEstimator: HeadPoseEstimator) -> None:
        """
        Set head pose estimator.

        Args:
            newEstimator: new estimator
        """
        self._headPoseEstimator = newEstimator

    @property
    def basicAttributesEstimator(self) -> BasicAttributesEstimator:
        """
        Get basic attributes estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            estimator
        """
        if self._basicAttributesEstimator is None:
            self._basicAttributesEstimator = self._faceEngine.createBasicAttributesEstimator()
        return self._basicAttributesEstimator

    @basicAttributesEstimator.setter
    def basicAttributesEstimator(self, newEstimator: BasicAttributesEstimator) -> None:
        """
        Set basic attributes estimator.

        Args:
            newEstimator: new estimator
        """
        self._basicAttributesEstimator = newEstimator

    @property
    def eyeEstimator(self) -> EyeEstimator:
        """
        Get eye estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            estimator
        """
        if self._eyeEstimator is None:
            self._eyeEstimator = self._faceEngine.createEyeEstimator()
        return self._eyeEstimator

    @eyeEstimator.setter
    def eyeEstimator(self, newEstimator: EyeEstimator) -> None:
        """
        Set eye estimator.

        Args:
            newEstimator: new estimator
        """
        self._eyeEstimator = newEstimator

    @property
    def emotionsEstimator(self) -> EmotionsEstimator:
        """
        Get emotions estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            estimator
        """
        if self._emotionsEstimator is None:
            self._emotionsEstimator = self._faceEngine.createEmotionEstimator()
        return self._emotionsEstimator

    @emotionsEstimator.setter
    def emotionsEstimator(self, newEstimator: EmotionsEstimator) -> None:
        """
        Set emotions estimator.

        Args:
            newEstimator: new estimator
        """
        self._emotionsEstimator = newEstimator

    @property
    def gazeDirectionEstimator(self) -> GazeEstimator:
        """
        Get gaze direction estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            estimator
        """
        if self._gazeDirectionEstimator is None:
            self._gazeDirectionEstimator = self._faceEngine.createGazeEstimator()
        return self._gazeDirectionEstimator

    @gazeDirectionEstimator.setter
    def gazeDirectionEstimator(self, newEstimator: GazeEstimator) -> None:
        """
        Set gaze direction estimator.

        Args:
            newEstimator: new estimator
        """
        self._gazeDirectionEstimator = newEstimator

    @property
    def mouthStateEstimator(self) -> MouthStateEstimator:
        """
        Get mouth state estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            estimator
        """
        if self._mouthStateEstimator is None:
            self._mouthStateEstimator = self._faceEngine.createMouthEstimator()
        return self._mouthStateEstimator

    @mouthStateEstimator.setter
    def mouthStateEstimator(self, newEstimator: MouthStateEstimator) -> None:
        """
        Set mouth state estimator.

        Args:
            newEstimator: new estimator
        """
        self._mouthStateEstimator = newEstimator

    @property
    def warpQualityEstimator(self) -> WarpQualityEstimator:
        """
        Get warp quality estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            estimator
        """
        if self._warpQualityEstimator is None:
            self._warpQualityEstimator = self._faceEngine.createWarpQualityEstimator()
        return self._warpQualityEstimator

    @warpQualityEstimator.setter
    def warpQualityEstimator(self, newEstimator: WarpQualityEstimator) -> None:
        """
        Set warp quality estimator.

        Args:
            newEstimator: new estimator
        """
        self._warpQualityEstimator = newEstimator

    @property
    def faceEngine(self) -> VLFaceEngine:
        """
        Get current faceengine.

        Returns:
           self._faceEngine
        """
        return self._faceEngine

    @faceEngine.setter
    def faceEngine(self, newFaceEngine: VLFaceEngine) -> None:
        """
        Set new faceengine. All initialize estimators will be re-initialized.

        Args:
            newFaceEngine: new faceengine
        """
        self._faceEngine = newFaceEngine
        for estimatorName in self.__slots__:
            if estimatorName == "_faceEngine":
                continue
            if getattr(self, estimatorName) is not None:
                self.initEstimator(FaceEstimatorsCollection._getEstimatorByAttributeName(estimatorName))
        self.warper = self._faceEngine.createWarper()

    def removeEstimator(self, estimator: FaceEstimator) -> None:
        """
        Remove estimators.

        Args:
            estimator: estimator for removing
        """
        estimatorName = self._getAttributeNameByEstimator(estimator)
        setattr(self, estimatorName, None)


FACE_ESTIMATORS_COLLECTION = FaceEstimatorsCollection()
