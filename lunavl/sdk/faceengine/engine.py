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
from ..estimators.body_estimators.body_attributes import BodyAttributesEstimator
from ..estimators.body_estimators.human_descriptor import HumanDescriptorEstimator
from ..estimators.body_estimators.humanwarper import HumanWarper
from ..estimators.face_estimators.ags import AGSEstimator
from ..estimators.face_estimators.background import FaceDetectionBackgroundEstimator
from ..estimators.face_estimators.basic_attributes import BasicAttributesEstimator
from ..estimators.face_estimators.credibility import CredibilityEstimator
from ..estimators.face_estimators.emotions import EmotionsEstimator
from ..estimators.face_estimators.eyebrow_expressions import EyebrowExpressionEstimator
from ..estimators.face_estimators.eyes import EyeEstimator, GazeEstimator
from ..estimators.face_estimators.face_descriptor import FaceDescriptorEstimator
from ..estimators.face_estimators.facewarper import FaceWarper
from ..estimators.face_estimators.fisheye import FisheyeEstimator
from ..estimators.face_estimators.glasses import GlassesEstimator
from ..estimators.face_estimators.head_pose import HeadPoseEstimator
from ..estimators.face_estimators.headwear import HeadwearEstimator
from ..estimators.face_estimators.image_type import ImageColorTypeEstimator
from ..estimators.face_estimators.livenessv1 import LivenessV1Estimator
from ..estimators.face_estimators.mask import MaskEstimator
from ..estimators.face_estimators.mouth_state import MouthStateEstimator
from ..estimators.face_estimators.natural_light import FaceNaturalLightEstimator
from ..estimators.face_estimators.red_eye import RedEyesEstimator
from ..estimators.face_estimators.warp_quality import WarpQualityEstimator
from ..estimators.image_estimators.orientation_mode import OrientationModeEstimator
from ..faceengine.setting_provider import DetectorType, FaceEngineSettingsProvider, RuntimeSettingsProvider
from ..globals import DEFAULT_HUMAN_DESCRIPTOR_VERSION as DHDV
from ..indexes.builder import IndexBuilder
from ..launch_options import DeviceClass, LaunchOptions


def _getLaunchOptions(launchOptions: Optional[LaunchOptions]) -> LaunchOptions:
    if not launchOptions:
        launchOptions = LaunchOptions()
    return launchOptions


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

    def createFaceDetector(
        self, detectorType: DetectorType, launchOptions: Optional[LaunchOptions] = None
    ) -> FaceDetector:
        """
        Create face detector.

        Args:
            detectorType: detector type.
            launchOptions: estimator launch options

        Returns:
            detector
        """
        if not launchOptions:
            launchOptions = LaunchOptions()
        return FaceDetector(
            self._faceEngine.createDetector(
                detectorType.coreDetectorType, launchOptions=launchOptions.coreLaunchOptions
            ),
            detectorType,
            launchOptions,
        )

    def createHeadPoseEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> HeadPoseEstimator:
        """
        Create head pose estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        if not launchOptions:
            launchOptions = LaunchOptions(DeviceClass.gpu)
        return HeadPoseEstimator(
            self._faceEngine.createHeadPoseEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createWarpQualityEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> WarpQualityEstimator:
        """
        Create an image quality estimator

        Args:
            launchOptions: estimator launch options.

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return WarpQualityEstimator(
            self._faceEngine.createQualityEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createFaceWarper(self) -> FaceWarper:
        """
        Create face warper, `see <warping.html>`_:

        Returns:
            warper.
        """
        return FaceWarper(self._faceEngine.createWarper())

    def createEmotionEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> EmotionsEstimator:
        """
        Create emotions estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return EmotionsEstimator(
            self._faceEngine.createEmotionsEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createMouthEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> MouthStateEstimator:
        """
        Create mouth state estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return MouthStateEstimator(
            self._faceEngine.createMouthEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createEyeEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> EyeEstimator:
        """
        Create eyes estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return EyeEstimator(
            self._faceEngine.createEyeEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createGazeEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> GazeEstimator:
        """
        Create gaze direction estimator.

        Args:
            launchOptions: estimator launch options.

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return GazeEstimator(
            self._faceEngine.createGazeEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createBasicAttributesEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> BasicAttributesEstimator:
        """
        Create basic attributes estimator (age, gender, ethnicity).

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return BasicAttributesEstimator(
            self._faceEngine.createAttributeEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createAGSEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> AGSEstimator:
        """
        Approximate garbage score estimator

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return AGSEstimator(
            self._faceEngine.createAGSEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createFaceDescriptorEstimator(
        self, descriptorVersion: int = 0, launchOptions: Optional[LaunchOptions] = None
    ) -> FaceDescriptorEstimator:
        """
        Approximate garbage score estimator

        Args:
            descriptorVersion: descriptor version to init estimator for or zero for use default descriptor version
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return FaceDescriptorEstimator(
            self._faceEngine.createExtractor(descriptorVersion, launchOptions=launchOptions.coreLaunchOptions),
            self.createFaceDescriptorFactory(descriptorVersion),
            launchOptions,
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

    def createHumanDetector(self, launchOptions: Optional[LaunchOptions] = None) -> HumanDetector:
        """
        Create human detector.

        Args:
            launchOptions: estimator launch options

        Returns:
            detector
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return HumanDetector(
            self._faceEngine.createHumanDetector(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

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

    def createHumanDescriptorEstimator(
        self, descriptorVersion: int = DHDV, launchOptions: Optional[LaunchOptions] = None
    ) -> HumanDescriptorEstimator:
        """
        Create human descriptor estimator.

        Args:
            launchOptions: estimator launch options
            descriptorVersion: human descriptor version

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return HumanDescriptorEstimator(
            self._faceEngine.createExtractor(descriptorVersion, launchOptions=launchOptions.coreLaunchOptions),
            self.createHumanDescriptorFactory(descriptorVersion),
            launchOptions,
        )

    def createMaskEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> MaskEstimator:
        """
        Create an medical mask estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return MaskEstimator(
            self._faceEngine.createMedicalMaskEstimator(launchOptions=launchOptions.coreLaunchOptions),
            launchOptions,
        )

    def createGlassesEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> GlassesEstimator:
        """
        Create a glasses estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return GlassesEstimator(
            self._faceEngine.createGlassesEstimator(launchOptions=launchOptions.coreLaunchOptions),
            launchOptions,
        )

    def createLivenessV1Estimator(self, launchOptions: Optional[LaunchOptions] = None) -> LivenessV1Estimator:
        """
        Create an one shot liveness estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return LivenessV1Estimator(
            self._faceEngine.createLivenessOneShotRGBEstimator(launchOptions=launchOptions.coreLaunchOptions),
            launchOptions,
        )

    def createOrientationModeEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> OrientationModeEstimator:
        """
        Create an orientation mode estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return OrientationModeEstimator(
            self._faceEngine.createOrientationEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createIndexBuilder(self, descriptorVersion: int = 0, capacity: int = 0) -> IndexBuilder:
        """
        Create an index builder for face
        Args:
            descriptorVersion: descriptor version, or zero if default should be used
            capacity: index capacity, or zero if default should be used

        Returns:
            index builder
        """
        return IndexBuilder(self._faceEngine, descriptorVersion=descriptorVersion, capacity=capacity)

    def createCredibilityEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> CredibilityEstimator:
        """
        Create a credibility estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return CredibilityEstimator(
            self._faceEngine.createCredibilityCheckEstimator(launchOptions=launchOptions.coreLaunchOptions),
            launchOptions,
        )

    def createEyebrowExpressionEstimator(
        self, launchOptions: Optional[LaunchOptions] = None
    ) -> EyebrowExpressionEstimator:
        """
        Create a eyebrow expression estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return EyebrowExpressionEstimator(
            self._faceEngine.createEyeBrowEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createHeadwearEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> HeadwearEstimator:
        """
        Create a headwear estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return HeadwearEstimator(
            self._faceEngine.createHeadWearEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createFaceNaturalLightEstimator(
        self, launchOptions: Optional[LaunchOptions] = None
    ) -> FaceNaturalLightEstimator:
        """
        Create a face natural light estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return FaceNaturalLightEstimator(
            self._faceEngine.createNaturalLightEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createRedEyeEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> RedEyesEstimator:
        """
        Create a red-eye estimator.

        Args:
            launchOptions: estimator launch options

        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return RedEyesEstimator(
            self._faceEngine.createRedEyeEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createFisheyeEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> FisheyeEstimator:
        """
        Create a fisheye effect estimator.

        Args:
            launchOptions: estimator launch options
        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return FisheyeEstimator(
            self._faceEngine.createFishEyeEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createFaceDetectionBackgroundEstimator(
        self, launchOptions: Optional[LaunchOptions] = None
    ) -> FaceDetectionBackgroundEstimator:
        """
        Create a face background estimator.

        Args:
            launchOptions: estimator launch options
        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return FaceDetectionBackgroundEstimator(
            self._faceEngine.createBackgroundEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createImageColorTypeEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> ImageColorTypeEstimator:
        """
        Create a image color type estimator.

        Args:
            launchOptions: estimator launch options
        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return ImageColorTypeEstimator(
            self._faceEngine.createBlackWhiteEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )

    def createBodyAttributesEstimator(self, launchOptions: Optional[LaunchOptions] = None) -> BodyAttributesEstimator:
        """
        Create a body attributes estimator.

        Args:
            launchOptions: estimator launch options
        Returns:
            estimator
        """
        launchOptions = _getLaunchOptions(launchOptions)
        return BodyAttributesEstimator(
            self._faceEngine.createHumanAttributeEstimator(launchOptions=launchOptions.coreLaunchOptions), launchOptions
        )
