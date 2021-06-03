"""Module contains face estimator collections.
"""
from enum import Enum
from typing import List, Optional, Union

from .estimators.face_estimators.ags import AGSEstimator
from .estimators.face_estimators.basic_attributes import BasicAttributesEstimator
from .estimators.face_estimators.emotions import EmotionsEstimator
from .estimators.face_estimators.eyes import EyeEstimator, GazeEstimator
from .estimators.face_estimators.face_descriptor import FaceDescriptorEstimator
from .estimators.face_estimators.head_pose import HeadPoseEstimator
from .estimators.face_estimators.mouth_state import MouthStateEstimator
from .estimators.face_estimators.warp_quality import WarpQualityEstimator
from .estimators.face_estimators.mask import MaskEstimator
from .estimators.face_estimators.glasses import GlassesEstimator
from .estimators.face_estimators.trustworthiness import TrustworthinessEstimator
from .estimators.face_estimators.facewarper import FaceWarper
from .estimators.face_estimators.livenessv1 import LivenessV1Estimator
from .estimators.image_estimators.orientation_mode import OrientationModeEstimator
from .faceengine.engine import VLFaceEngine


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
    #: ags estimator
    AGS = 8
    #: face descriptor estimator
    Descriptor = 9
    #: mask estimator
    Mask = 10
    #: glasses estimator
    Glasses = 11
    #: liveness v1 estimator
    LivenessV1 = 12
    #: orientation mode estimator
    OrientationMode = 13
    #: trustworthiness estimator
    Trustworthiness = 14


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
        _AGSEstimator (Optional[AGSEstimator]): lazy load ags estimator
        _descriptorEstimator (Optional[FaceDescriptorEstimator]): lazy load face descriptor estimator
        _maskEstimator (Optional[MaskEstimator]): lazy mask estimator
        _glassesEstimator (Optional[GlassesEstimator]): lazy glasses estimator
        _livenessV1Estimator (Optional[LivenessV1Estimator]): lazy livenessv1 estimator
        _orientationModeEstimator (Optional[OrientationModeEstimator]): lazy orientation mode estimator
        _trustworthinessEstimator (Optional[TrustworthinessEstimator]): lazy trustworthiness estimator
        warper (Optional[Warper]): warper
    """

    __slots__ = (
        "_headPoseEstimator",
        "_eyeEstimator",
        "_gazeDirectionEstimator",
        "_mouthStateEstimator",
        "_warpQualityEstimator",
        "_basicAttributesEstimator",
        "_emotionsEstimator",
        "_faceEngine",
        "_AGSEstimator",
        "warper",
        "_descriptorEstimator",
        "_maskEstimator",
        "_glassesEstimator",
        "_livenessV1Estimator",
        "_orientationModeEstimator",
        "_trustworthinessEstimator",
    )

    def __init__(
        self, startEstimators: Optional[List[FaceEstimator]] = None, faceEngine: Optional[VLFaceEngine] = None
    ):
        """
        Init.

        Args:
            startEstimators: list of estimators which will be initiate now
            faceEngine: faceengine, factory for estimators
        """
        if faceEngine is None:
            self._faceEngine = VLFaceEngine()
        else:
            self._faceEngine = faceEngine

        self._basicAttributesEstimator: Union[None, BasicAttributesEstimator] = None
        self._eyeEstimator: Union[None, EyeEstimator] = None
        self._emotionsEstimator: Union[None, EmotionsEstimator] = None
        self._gazeDirectionEstimator: Union[None, GazeEstimator] = None
        self._mouthStateEstimator: Union[None, MouthStateEstimator] = None
        self._warpQualityEstimator: Union[None, WarpQualityEstimator] = None
        self._headPoseEstimator: Union[None, HeadPoseEstimator] = None
        self._AGSEstimator: Union[None, AGSEstimator] = None
        self._descriptorEstimator: Union[None, FaceDescriptorEstimator] = None
        self._maskEstimator: Union[None, MaskEstimator] = None
        self._glassesEstimator: Union[None, GlassesEstimator] = None
        self._livenessV1Estimator: Union[None, LivenessV1Estimator] = None
        self._orientationModeEstimator: Union[None, OrientationModeEstimator] = None
        self._trustworthinessEstimator: Union[None, TrustworthinessEstimator] = None
        self.warper: FaceWarper = self._faceEngine.createFaceWarper()

        if startEstimators:
            for estimator in set(startEstimators):
                self.initEstimator(estimator)

    @staticmethod
    def _getEstimatorByAttributeName(estimatorAttributeName: str) -> FaceEstimator:
        """
        Get face estimator by attribute name which corresponding to estimator

        Args:
            estimatorAttributeName:  attribute estimator name

        Returns:
            corresponding face estimator

        Raises:
            ValueError: if face estimator not found
        """
        for estimator in FaceEstimator:
            if estimator.name.lower() == estimatorAttributeName[1 : -len("Estimator")]:  # noqa: E203
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
            ValueError: if attribute name not found
        """
        for estimatorName in self.__slots__:
            if estimatorName == "_faceEngine":
                continue
            if estimator.name.lower() == estimatorName[1 : -len("Estimator")]:  # noqa: E203
                return estimatorName
        raise ValueError("Bad estimator")

    def initEstimator(self, estimator: FaceEstimator) -> None:
        """
        Create an estimator. Create new estimator with help self._faceengine

        Args:
            estimator: estimator for creating
        Raises:
            ValueError: if estimator not found
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
        elif estimator == FaceEstimator.AGS:
            self._AGSEstimator = self._faceEngine.createAGSEstimator()
        elif estimator == FaceEstimator.Descriptor:
            self._descriptorEstimator = self._faceEngine.createFaceDescriptorEstimator()
        elif estimator == FaceEstimator.Mask:
            self._maskEstimator = self._faceEngine.createMaskEstimator()
        elif estimator == FaceEstimator.Glasses:
            self._glassesEstimator = self._faceEngine.createGlassesEstimator()
        elif estimator == FaceEstimator.LivenessV1:
            self._livenessV1Estimator = self._faceEngine.createLivenessV1Estimator()
        elif estimator == FaceEstimator.OrientationMode:
            self._orientationModeEstimator = self._faceEngine.createOrientationModeEstimator()
        elif estimator == FaceEstimator.Trustworthiness:
            self._trustworthinessEstimator = self._faceEngine.createTrustworthinessEstimator()
        else:
            raise ValueError("Bad estimator type")

    @property
    def descriptorEstimator(self) -> FaceDescriptorEstimator:
        """
        Get head pose estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            estimator
        """
        if self._descriptorEstimator is None:
            self._descriptorEstimator = self._faceEngine.createFaceDescriptorEstimator()
        return self._descriptorEstimator

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

    # pylint: disable=C0103
    @property
    def AGSEstimator(self) -> AGSEstimator:  # type: ignore
        """
        Get ags estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            estimator
        """
        if self._AGSEstimator is None:
            self._AGSEstimator = self._faceEngine.createAGSEstimator()
        return self._AGSEstimator

    # pylint: disable=C0103
    @AGSEstimator.setter
    def AGSEstimator(self, newEstimator: AGSEstimator) -> None:  # type: ignore
        """
        Set ags estimator.

        Args:
            newEstimator: new estimator
        """
        self._AGSEstimator = newEstimator

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
    def maskEstimator(self) -> MaskEstimator:
        """
        Get mask estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            mask estimator
        """
        if self._maskEstimator is None:
            self._maskEstimator = self._faceEngine.createMaskEstimator()
        return self._maskEstimator

    @maskEstimator.setter
    def maskEstimator(self, newEstimator: MaskEstimator) -> None:
        """
        Set warp mask estimator.
        Args:
            newEstimator: new mask estimator
        """
        self._maskEstimator = newEstimator

    @property
    def glassesEstimator(self) -> GlassesEstimator:
        """
        Get glasses estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            glasses estimator
        """
        if self._glassesEstimator is None:
            self._glassesEstimator = self._faceEngine.createGlassesEstimator()
        return self._glassesEstimator

    @glassesEstimator.setter
    def glassesEstimator(self, newEstimator: GlassesEstimator) -> None:
        """
        Set warp glasses estimator.
        Args:
            newEstimator: new glasses estimator
        """
        self._glassesEstimator = newEstimator

    @property
    def livenessV1Estimator(self) -> LivenessV1Estimator:
        """
        LivenessV1 estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            estimator
        """
        if self._livenessV1Estimator is None:
            self._livenessV1Estimator = self._faceEngine.createLivenessV1Estimator()
        return self._livenessV1Estimator

    @livenessV1Estimator.setter
    def livenessV1Estimator(self, newEstimator: LivenessV1Estimator) -> None:
        """
        Set livenessV1 estimator.

        Args:
            newEstimator: new estimator
        """
        self._livenessV1Estimator = newEstimator

    @property
    def orientationModeEstimator(self) -> OrientationModeEstimator:
        """
        Orientation mode estimator.

        If estimator is initialized it will be returned otherwise it will be initialized and returned

        Returns:
            estimator
        """
        if self._orientationModeEstimator is None:
            self._orientationModeEstimator = self._faceEngine.createOrientationModeEstimator()
        return self._orientationModeEstimator

    @orientationModeEstimator.setter
    def orientationModeEstimator(self, newEstimator: OrientationModeEstimator) -> None:
        """
        Set orientation mode estimator
        Args:
            newEstimator: new estimator
        """
        self._orientationModeEstimator = newEstimator

    @property
    def trustworthinessEstimator(self) -> TrustworthinessEstimator:
        """
        Get trustworthiness estimator.
        Returns:
            trustworthiness estimator
        """
        if self._trustworthinessEstimator is None:
            self._trustworthinessEstimator = self._faceEngine.createTrustworthinessEstimator()
        return self._trustworthinessEstimator

    @trustworthinessEstimator.setter
    def trustworthinessEstimator(self, newEstimator: TrustworthinessEstimator) -> None:
        """
        Set warp trustworthiness estimator.
        Args:
            newEstimator: new trustworthiness estimator
        """
        self._trustworthinessEstimator = newEstimator

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
        self.warper = self._faceEngine.createFaceWarper()

    def removeEstimator(self, estimator: FaceEstimator) -> None:
        """
        Remove estimators.

        Args:
            estimator: estimator for removing
        """
        estimatorName = self._getAttributeNameByEstimator(estimator)
        setattr(self, estimatorName, None)
