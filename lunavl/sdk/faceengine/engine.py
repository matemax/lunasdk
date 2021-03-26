"""Module realize wraps on facengine objects
"""
import os
from pathlib import Path
from typing import Optional, Union

import FaceEngine as CoreFE  # pylint: disable=E0611,E0401

from ..descriptors.descriptors import FaceDescriptorFactory, HumanDescriptorFactory
from ..descriptors.matcher import FaceMatcher
from ..detectors.facedetector import FaceDetector
from ..detectors.humandetector import HumanDetector
from ..estimators.body_estimators.human_descriptor import HumanDescriptorEstimator
from ..estimators.body_estimators.humanwarper import HumanWarper
from ..estimators.face_estimators.ags import AGSEstimator
from ..estimators.face_estimators.basic_attributes import BasicAttributesEstimator
from ..estimators.face_estimators.emotions import EmotionsEstimator
from ..estimators.face_estimators.eyes import EyeEstimator, GazeEstimator
from ..estimators.face_estimators.face_descriptor import FaceDescriptorEstimator
from ..estimators.face_estimators.facewarper import FaceWarper
from ..estimators.face_estimators.glasses import GlassesEstimator
from ..estimators.face_estimators.head_pose import HeadPoseEstimator
from ..estimators.face_estimators.livenessv1 import LivenessV1Estimator
from ..estimators.face_estimators.mask import MaskEstimator
from ..estimators.face_estimators.mouth_state import MouthStateEstimator
from ..estimators.face_estimators.warp_quality import WarpQualityEstimator
from ..estimators.image_estimators.orientation_mode import OrientationModeEstimator
from ..faceengine.setting_provider import DetectorType, FaceEngineSettingsProvider, RuntimeSettingsProvider
from ..globals import DEFAULT_HUMAN_DESCRIPTOR_VERSION as DHDV
from ..indexes.builder import IndexBuilder


class VLFaceEngine:
    """
    Wraper on FaceEngine.

    Attributes:
        dataPath (str): path to a faceengine data folder
        faceEngineProvider (FaceEngineSettingsProvider): face engine settings provider
        runtimeProvider (RuntimeSettingsProvider): runtime settings provider
        _faceEngine (PyIFaceEngine): python C++ binding on IFaceEngine, Root LUNA SDK object interface
    """

    # path to a file with license info
    license: Optional[Union[str, Path]] = None

    def __init__(
        self,
        pathToData: Optional[str] = None,
        faceEngineConf: Optional[Union[str, FaceEngineSettingsProvider]] = None,
        runtimeConf: Optional[Union[str, RuntimeSettingsProvider]] = None,
    ):
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
        self._faceEngine = CoreFE.createFaceEngine(
            dataPath=pathToData, configPath=str(self.faceEngineProvider.pathToConfig)
        )
        if self.license:
            self.activate(self.license)

        self._faceEngine.setSettingsProvider(self.faceEngineProvider.coreProvider)
        self._faceEngine.setRuntimeSettingsProvider(self.runtimeProvider.coreProvider)

    def activate(self, pathToLicense: Union[str, Path]):
        """
        Activate license
        Args:
            pathToLicense: path to the file with license info
        """
        _license = self._faceEngine.getLicense()
        return self._faceEngine.activateLicense(
            _license, pathToLicense if isinstance(pathToLicense, str) else (str(pathToLicense))
        )

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

    def createFaceWarper(self) -> FaceWarper:
        """
        Create face warper, `see <warping.html>`_:

        Returns:
            warper.
        """
        return FaceWarper(self._faceEngine.createWarper())

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
        return MouthStateEstimator(self._faceEngine.createMouthEstimator())

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

    def createFaceDescriptorEstimator(self, descriptorVersion: int = 0) -> FaceDescriptorEstimator:
        """
        Approximate garbage score estimator

        Args:
            descriptorVersion: descriptor version to init estimator for or zero for use default descriptor version

        Returns:
            estimator
        """
        return FaceDescriptorEstimator(
            self._faceEngine.createExtractor(descriptorVersion), self.createFaceDescriptorFactory(descriptorVersion)
        )

    def createFaceDescriptorFactory(self, descriptorVersion: int = 0) -> FaceDescriptorFactory:
        """
        Create face descriptor factory
        Args:
            descriptorVersion: descriptor version or zero for use default descriptor version

        Returns:
            face descriptor factory
        """
        return FaceDescriptorFactory(self, descriptorVersion=descriptorVersion)

    def createFaceMatcher(self, descriptorVersion: int = 0) -> FaceMatcher:
        """
        Create face matcher
        Args:
            descriptorVersion: descriptor version or zero for use default descriptor version

        Returns:
            face matcher
        """
        return FaceMatcher(self._faceEngine.createMatcher(descriptorVersion), self.createFaceDescriptorFactory())

    @property
    def coreFaceEngine(self) -> CoreFE.PyIFaceEngine:
        """
        Get core face engine

        Returns:
            core face engine
        """
        return self._faceEngine

    def createHumanDetector(self) -> HumanDetector:
        """
        Create human detector.
        Returns:
            detector
        """
        return HumanDetector(self._faceEngine.createHumanDetector())

    def createHumanWarper(self) -> HumanWarper:
        """
        Create human body warper.

        Returns:
            warper.

        """
        return HumanWarper(self._faceEngine.createHumanWarper())

    def createHumanDescriptorFactory(self, descriptorVersion: int = DHDV) -> HumanDescriptorFactory:
        """
        Create human descriptor factory

        Args:
            descriptorVersion: descriptor version to init estimator for or zero for use default descriptor version

        Returns:
            human descriptor factory
        """
        return HumanDescriptorFactory(self, descriptorVersion=descriptorVersion)

    def createHumanDescriptorEstimator(self, descriptorVersion: int = DHDV) -> HumanDescriptorEstimator:
        """
        Create human descriptor estimator

        Returns:
            estimator
        """
        return HumanDescriptorEstimator(
            self._faceEngine.createExtractor(descriptorVersion), self.createHumanDescriptorFactory(descriptorVersion)
        )

    def createMaskEstimator(self) -> MaskEstimator:
        """
        Create an medical mask estimator

        Returns:
            estimator
        """
        return MaskEstimator(self._faceEngine.createMedicalMaskEstimator())

    def createGlassesEstimator(self) -> GlassesEstimator:
        """
        Create a glasses estimator.

        Returns:
            estimator
        """
        return GlassesEstimator(self._faceEngine.createGlassesEstimator())

    def createLivenessV1Estimator(
        self,
    ) -> LivenessV1Estimator:
        """
        Create an one shot liveness estimator.
        Returns:
            estimator
        """
        return LivenessV1Estimator(self._faceEngine.createLivenessOneShotRGBEstimator())

    def createOrientationModeEstimator(self) -> OrientationModeEstimator:
        """
        Create an orientation mode estimator

        Returns:
            estimator
        """
        return OrientationModeEstimator(self._faceEngine.createOrientationEstimator())

    def createIndexBuilder(self) -> IndexBuilder:
        """
        Create an index builder for face
        Returns:
            index builder
        """
        return IndexBuilder(self._faceEngine)
