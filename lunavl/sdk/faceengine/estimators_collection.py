from enum import Enum

from typing import List, Optional

from lunavl.sdk.faceengine.engine import VLFaceEngine, FACE_ENGINE


class FaceEstimator(Enum):
    HeadPose = 1
    Eye = 2
    Emotions = 3
    BasicAttributes = 4
    GazeDirection = 5
    MouthState = 6
    WarpQuality = 7


class FaceEstimatorsCollection:
    __slots__ = ("_headPoseEstimator", "_eyeEstimator", "_gazeDirectionEstimator", "_mouthStateEstimator",
                 "_warpQualityEstimator", "_basicAttributesEstimator", "_emotionsEstimator", "_faceEngine")

    def __init__(self, startEstimators: Optional[List[FaceEstimator]] = None,
                 faceEngine: Optional[VLFaceEngine] = None):
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

        if startEstimators:
            for estimator in set(startEstimators):
                self.initEstimator(estimator)

    @staticmethod
    def _getEstimatorByAttributeName(attributeName: str):
        for estimator in FaceEstimator:
            if estimator.name.lower() == attributeName[1: -len("Estimator")]:
                return estimator
        raise ValueError("Bad attribute name")

    def _getAttributeNameByEstimator(self, estimator: FaceEstimator):
        for estimatorName in self.__slots__:
            if estimatorName == "_faceEngine":
                continue
            if estimator.name.lower() == estimatorName[1: -len("Estimator")]:
                return estimatorName
        raise ValueError("Bad estimator")

    def initEstimator(self, estimator: FaceEstimator):
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
    def headPoseEstimator(self):
        if self._headPoseEstimator is None:
            self._headPoseEstimator = self._faceEngine.createHeadPoseEstimator()
        return self._headPoseEstimator

    @headPoseEstimator.setter
    def headPoseEstimator(self, newEstimator):
        self._headPoseEstimator = newEstimator

    @property
    def basicAttributesEstimator(self):
        if self._basicAttributesEstimator is None:
            self._basicAttributesEstimator = self._faceEngine.createBasicAttributesEstimator()
        return self._basicAttributesEstimator

    @basicAttributesEstimator.setter
    def basicAttributesEstimator(self, newEstimator):
        self._basicAttributesEstimator = newEstimator

    @property
    def eyeEstimator(self):
        if self._eyeEstimator is None:
            self._eyeEstimator = self._faceEngine.createEyeEstimator()
        return self._eyeEstimator

    @eyeEstimator.setter
    def eyeEstimator(self, newEstimator):
        self._eyeEstimator = newEstimator

    @property
    def emotionsEstimator(self):
        if self._emotionsEstimator is None:
            self._emotionsEstimator = self._faceEngine.createEmotionEstimator()
        return self._emotionsEstimator

    @emotionsEstimator.setter
    def emotionsEstimator(self, newEstimator):
        self._emotionsEstimator = newEstimator

    @property
    def gazeDirectionEstimator(self):
        if self._gazeDirectionEstimator is None:
            self._gazeDirectionEstimator = self._faceEngine.createGazeEstimator()
        return self._gazeDirectionEstimator

    @gazeDirectionEstimator.setter
    def gazeDirectionEstimator(self, newEstimator):
        self._gazeDirectionEstimator = newEstimator

    @property
    def mouthStateEstimator(self):
        if self._mouthStateEstimator is None:
            self._mouthStateEstimator = self._faceEngine.createMouthEstimator()
        return self._mouthStateEstimator

    @mouthStateEstimator.setter
    def mouthStateEstimator(self, newEstimator):
        self._mouthStateEstimator = newEstimator

    @property
    def warpQualityEstimator(self):
        if self._warpQualityEstimator is None:
            self._warpQualityEstimator = self._faceEngine.createWarpQualityEstimator()
        return self._warpQualityEstimator

    @warpQualityEstimator.setter
    def warpQualityEstimator(self, newEstimator):
        self._warpQualityEstimator = newEstimator

    @property
    def faceEngine(self):
        return self._faceEngine

    @faceEngine.setter
    def faceEngine(self, newFaceEngine):
        self._faceEngine = newFaceEngine
        for estimatorName in self.__slots__:
            if estimatorName == "_faceEngine":
                continue
            if getattr(self, estimatorName) is not None:
                self.initEstimator(FaceEstimatorsCollection._getEstimatorByAttributeName(estimatorName))

    def removeEstimator(self, estimator: FaceEstimator) -> None:
        estimatorName = self._getAttributeNameByEstimator(estimator)
        setattr(self, estimatorName, None)


FACE_ESTIMATORS_COLLECTION = FaceEstimatorsCollection()
