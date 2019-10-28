"""Module realize wraps on facengine objects
"""
import os
from typing import Optional, Union

import FaceEngine as CoreFE  # pylint: disable=E0611,E0401

from lunavl.sdk.estimators.face_estimators.ags import AGSEstimator
from lunavl.sdk.estimators.face_estimators.basic_attributes import BasicAttributesEstimator
from lunavl.sdk.estimators.face_estimators.emotions import EmotionsEstimator
from lunavl.sdk.estimators.face_estimators.eyes import EyeEstimator, GazeEstimator
from lunavl.sdk.estimators.face_estimators.face_descriptor import FaceDescriptorEstimator
from lunavl.sdk.estimators.face_estimators.mouth_state import MouthStateEstimator

from lunavl.sdk.estimators.face_estimators.warp_quality import WarpQualityEstimator
from lunavl.sdk.estimators.face_estimators.warper import Warper

from lunavl.sdk.estimators.face_estimators.head_pose import HeadPoseEstimator
from lunavl.sdk.faceengine.descriptors import FaceDescriptorFactory
from lunavl.sdk.faceengine.matcher import FaceMatcher
from lunavl.sdk.faceengine.setting_provider import DetectorType, FaceEngineSettingsProvider, RuntimeSettingsProvider

from ..faceengine.facedetector import FaceDetector


class VLFaceEngine:
    """
    Wraper on FaceEngine.

    Attributes:
        dataPath (str): path to a faceengine data folder
        faceEngineProvider (FaceEngineSettingsProvider): face engine settings provider
        runtimeProvider (RuntimeSettingsProvider): runtime settings provider
        _faceEngine (PyIFaceEngine): python C++ binding on IFaceEngine, Root LUNA SDK object interface
    """

    def __init__(self, pathToData: Optional[str] = None,
                 faceEngineConf: Optional[Union[str, FaceEngineSettingsProvider]] = None,
                 runtimeConf: Optional[Union[str, RuntimeSettingsProvider]] = None):
        """
        Init.

        Args:
            pathToData: path to a faceengine data folder
            faceEngineConf:  path to a faceengine configuration file
            runtimeConf:  path to a runtime configuration file
        """
        if pathToData is None:
            if "FSDK_ROOT" in os.environ:
                pathToData = os.path.join(os.environ["FSDK_ROOT"], "data")
            else:
                raise ValueError(
                    "Failed on path to faceengine luna data folder, set variable pathToData or set"
                    "environment variable *FSDK_ROOT*"
                )
        if faceEngineConf is None:
            self.faceEngineProvider = FaceEngineSettingsProvider()
        elif isinstance(faceEngineConf, str):
            self.faceEngineProvider = FaceEngineSettingsProvider(faceEngineConf)
        else:
            self.faceEngineProvider = faceEngineConf

        if runtimeConf is None:
            self.runtimeProvider = RuntimeSettingsProvider()
        elif isinstance(runtimeConf, str):
            self.runtimeProvider = RuntimeSettingsProvider(runtimeConf)
        else:
            self.runtimeProvider = runtimeConf

        self.dataPath = pathToData
        # todo: validate initialize
        self._faceEngine = CoreFE.createFaceEngine(dataPath=pathToData,
                                                   configPath=str(self.faceEngineProvider.pathToConfig))

        self._faceEngine.setSettingsProvider(self.faceEngineProvider.coreProvider)
        self._faceEngine.setRuntimeSettingsProvider(self.runtimeProvider.coreProvider)

    def createFaceDetector(self, detectorType: DetectorType) -> FaceDetector:
        """
        Create face detector.

        Args:
            detectorType: detector type

        Returns:
            detector
        """
        return FaceDetector(self._faceEngine.createDetector(detectorType.coreDetectorType), detectorType)

    def createHeadPoseEstimator(self) -> HeadPoseEstimator:
        """
        Create head pose estimator

        Returns:
            estimator
        """
        return HeadPoseEstimator(self._faceEngine.createHeadPoseEstimator())

    def createWarpQualityEstimator(self) -> WarpQualityEstimator:
        """
        Create an image quality estimator

        Returns:
            estimator
        """
        return WarpQualityEstimator(self._faceEngine.createQualityEstimator())

    def createWarper(self) -> Warper:
        """
        Create warper, `see <warping.html>`_:

        Returns:
            warper.
        """
        return Warper(self._faceEngine.createWarper())

    def createEmotionEstimator(self) -> EmotionsEstimator:
        """
        Create emotions estimator

        Returns:
            estimator
        """
        return EmotionsEstimator(self._faceEngine.createEmotionsEstimator())

    def createMouthEstimator(self) -> MouthStateEstimator:
        """
        Create mouth state estimator

        Returns:
            estimator
        """
        return MouthStateEstimator(self._faceEngine.createSmileEstimator())

    def createEyeEstimator(self) -> EyeEstimator:
        """
        Create eyes estimator

        Returns:
            estimator
        """
        return EyeEstimator(self._faceEngine.createEyeEstimator())

    def createGazeEstimator(self) -> GazeEstimator:
        """
        Create gaze direction estimator

        Returns:
            estimator
        """
        return GazeEstimator(self._faceEngine.createGazeEstimator())

    def createBasicAttributesEstimator(self) -> BasicAttributesEstimator:
        """
        Create basic attributes estimator (age, gender, ethnicity)

        Returns:
            estimator
        """
        return BasicAttributesEstimator(self._faceEngine.createAttributeEstimator())

    def createAGSEstimator(self) -> AGSEstimator:
        """
        Approximate garbage score estimator

        Returns:
            estimator
        """
        return AGSEstimator(self._faceEngine.createAGSEstimator())

    def createFaceDescriptorEstimator(self) -> FaceDescriptorEstimator:
        """
        Approximate garbage score estimator

        Returns:
            estimator
        """
        return FaceDescriptorEstimator(self._faceEngine.createExtractor(), self.createFaceDescriptorFactory())

    def createFaceDescriptorFactory(self) -> FaceDescriptorFactory:
        return FaceDescriptorFactory(self)

    def createFaceMatcher(self) -> FaceMatcher:
        return FaceMatcher(self._faceEngine.createMatcher(), self.createFaceDescriptorFactory())

    @property
    def coreFaceEngine(self) -> CoreFE.PyIFaceEngine:
        return self._faceEngine
