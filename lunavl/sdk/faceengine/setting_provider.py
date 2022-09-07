"""
SDK configuration module.
"""
import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple, Type, TypeVar, Union

import FaceEngine as CoreFE
from FaceEngine import ObjectDetectorClassType, PyISettingsProvider  # pylint: disable=E0611,E0401

from lunavl.sdk.launch_options import DeviceClass

BI_ENUM = TypeVar("BI_ENUM", bound="BiDirectionEnum")


class BiDirectionEnum(Enum):
    """
    Bi direction enum.
    """

    @classmethod
    def getEnum(cls: Type[BI_ENUM], enumValue: Union[int, str]) -> BI_ENUM:
        """
        Get enum by value.

        Args:
            enumValue: value

        Returns:
           element of the enum with value which is equal to the enumValue.
        Raises:
              KeyError: if element not found.
        """
        for enumMember in cls:
            if enumMember.value == enumValue:
                return enumMember
        raise KeyError("Enum {} does not contain  member with value {}".format(cls.__name__, enumValue))


class CpuClass(Enum):
    """Class of cpu by supported instructions"""

    auto = "auto"
    sse4 = "sse4"
    avx = "avx"
    avx2 = "avx2"
    arm = "arm"


class VerboseLogging(BiDirectionEnum):
    """
    Level of log versobing enum
    """

    error = 0
    warnings = 1
    info = 2
    debug = 3


class Distance(BiDirectionEnum):
    """
    Descriptor distance type enum.
    """

    l1 = "L1"
    l2 = "L2"


class NMS(Enum):
    """
    NMS type enum.
    """

    mean = "mean"
    best = "best"


class Point4:
    """
    Point in 4-dimensional space.
    Attributes:
        x (float): x coordinate
        y (float): y coordinate
        z (float): z coordinate
        w (float): w coordinate
    """

    def __init__(self, x: float, y: float, z: float, w: float):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def asTuple(self) -> Tuple[float, float, float, float]:
        """
        Convert point to tuple.

        Returns:
            tuple from coordinate
        """
        return self.x, self.y, self.z, self.w


class Point3:
    """
    Point in 3-dimensional space.
    Attributes:
        x (float): x coordinate
        y (float): y coordinate
        z (float): z coordinate
    """

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def asTuple(self) -> Tuple[float, float, float]:
        """
        Convert point to tuple.

        Returns:
            tuple from coordinate
        """
        return self.x, self.y, self.z


class Point2:
    """
    Point in 2-dimensional space.
    Attributes:
        x (float): x coordinate
        y (float): y coordinate
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def asTuple(self) -> Tuple[float, float]:
        """
        Convert point to tuple.

        Returns:
            tuple from coordinate
        """
        return self.x, self.y


class DetectorType(BiDirectionEnum):
    """
    Detector types enum
    """

    FACE_DET_DEFAULT = "Default"
    FACE_DET_V1 = "FaceDetV1"  #: todo description
    FACE_DET_V2 = "FaceDetV2"
    FACE_DET_V3 = "FaceDetV3"

    @property
    def coreDetectorType(self) -> ObjectDetectorClassType:
        """
        Convert  self to core detector type

        Returns:
            ObjectDetectorClassType
        """
        mapEnumToCoreEnum = {
            "Default": "FACE_DET_DEFAULT",
            "FaceDetV1": "FACE_DET_V1",
            "FaceDetV2": "FACE_DET_V2",
            "FaceDetV3": "FACE_DET_V3",
        }
        return getattr(ObjectDetectorClassType, mapEnumToCoreEnum[self.value])


class BaseSettingsSection:
    """
    Base class for a section of settings.

    Proxy model to core settings provider.

    Attributes:
        _coreSettingProvider (PyISettingsProvider): core settings faceEngineProvider
    """

    # (str): section name
    sectionName: str

    def __init__(self, coreSettingProvider: PyISettingsProvider):
        self._coreSettingProvider = coreSettingProvider

    def setValue(self, name: str, value: Any) -> None:
        """
        Set a value

        Args:
            name: setting name
            value: new value
        """
        self._coreSettingProvider.setValue(self.__class__.sectionName, name, CoreFE.SettingsProviderValue(value))

    def getValue(self, name: str) -> Optional[Any]:
        """
        Get setting value
        Args:
            name: setting name

        Returns:
            a value or None if settings does not exists
        """
        value = self._coreSettingProvider.getValue(self.__class__.sectionName, name)
        if value == []:
            return None
        return value[0]


class SystemSettings(BaseSettingsSection):
    """
    Common system settings.

    Properties:
        - verboseLogging (VerboseLogging): Level of log verbosing
        - betaMode (bool): enable experimental features.
        - defaultDetectorType (DetectorType): default detector type
    """

    sectionName = "system"

    @property
    def verboseLogging(self) -> Optional[VerboseLogging]:
        """
        Getter for verboseLogging

        Returns:
            verboseLogging
        """
        value = self.getValue("verboseLogging")
        if value is None:
            return None
        return VerboseLogging.getEnum(value)

    @verboseLogging.setter
    def verboseLogging(self, value: VerboseLogging) -> None:
        """
        Setter for cpuClass.

        Args:
            value: new value
        """
        self.setValue("verboseLogging", value.value)

    @property
    def betaMode(self) -> Optional[bool]:
        """
        Getter for betaMode

        Returns:
            betaMode
        """
        value = self.getValue("betaMode")
        if value is None:
            return None
        return bool(value)

    @betaMode.setter
    def betaMode(self, value: bool) -> None:
        """
        Setter for betaMode
        Args:
            value: new value
        """
        self.setValue("betaMode", int(value))

    @property
    def defaultDetectorType(self) -> Optional[DetectorType]:
        """
        Getter for defaultDetectorType

        Returns:
            default detector type
        """
        value = self.getValue("defaultDetectorType")
        if value is None:
            return None
        return DetectorType.getEnum(value)

    @defaultDetectorType.setter
    def defaultDetectorType(self, value: DetectorType) -> None:
        """
        Setter for defaultDetectorType
        Args:
                value: new value
        """
        self.setValue("defaultDetectorType", value.value)


class RuntimeSettings(BaseSettingsSection):
    """
    Flower library is the default neural network inference engine.
    The library is used for:

        - face detectors;
        - estimators(attributes, quality);
        - face descriptors

    Properties:
        cpuClass (CpuClass): class of cpu by supported instructions
        deviceClass (DeviceClass):  execution device type cpu, gpu.
        numThreads (int): number of worker threads.
        verboseLogging (VerboseLogging): level of verbose logging
        numComputeStreams (int):  increases performance, but works only with new versions of nvidia drivers
    """

    sectionName = "Runtime"

    @property
    def deviceClass(self) -> Optional[DeviceClass]:
        """
        Get device class.

        Returns:
            device class
        """
        value = self.getValue("deviceClass")
        if value is None:
            return None
        return DeviceClass[value]

    @deviceClass.setter
    def deviceClass(self, value: DeviceClass) -> None:
        """
        Setter for deviceClass
        Args:
            value: new value
        """
        self.setValue("deviceClass", value.value)

    @property
    def cpuClass(self) -> Optional[CpuClass]:
        """
        Getter for cpuClass

        Returns:
            cpuClass
        """
        value = self.getValue("cpuClass")
        if value is None:
            return None
        return CpuClass[value]

    @cpuClass.setter
    def cpuClass(self, value: CpuClass) -> None:
        """
        Setter for cpuClass.

        Args:
            value: new value
        """
        self.setValue("cpuClass", value.value)

    @property
    def numThreads(self) -> Optional[int]:
        """
        Getter for numThreads

        Returns:
            numThreads
        """
        return self.getValue("numThreads")

    @numThreads.setter
    def numThreads(self, value: int) -> None:
        """
        Setter for numThreads
        Args:
            value: new value
        """
        self.setValue("numThreads", value)

    @property
    def verboseLogging(self) -> Optional[VerboseLogging]:
        """
        Getter for verboseLogging

        Returns:
            verboseLogging
        """
        value = self.getValue("verboseLogging")
        if value is None:
            return None
        return VerboseLogging.getEnum(value)

    @verboseLogging.setter
    def verboseLogging(self, value: VerboseLogging) -> None:
        """
        Setter for verboseLogging
        Args:
            value: new value
        """
        self.setValue("verboseLogging", value.value)

    @property
    def numComputeStreams(self) -> Optional[int]:
        """
        Getter for numComputeStreams

        Returns:
            numComputeStreams
        """
        return self.getValue("numComputeStreams")

    @numComputeStreams.setter
    def numComputeStreams(self, value: int) -> None:
        """
        Setter for numComputeStreams
        Args:
            value: new value
        """
        self.setValue("numComputeStreams", value)


class DescriptorFactorySettings(BaseSettingsSection):
    """
    Descriptor factory settings.

    Properties:

        - model (int): CNN face descriptor version.
        - UseMobileNet (bool): mobile Net is faster but less accurate
        - distance (Distance): distance between descriptors on matching. L1 faster,L2 make better precision.
        - descriptorCountWarningLevel (float): Threshold,that limits the ratio of created  descriptors to the amount,
            defined by your liscence. Warning Level When the threshold is exceeded, FSDK prints the warning.

    """

    sectionName = "DescriptorFactory::Settings"

    @property
    def model(self) -> Optional[int]:
        """
        Getter for model

        Returns:
            model
        """
        return self.getValue("model")

    @model.setter
    def model(self, value: int) -> None:
        """
        Setter for model
        Args:
            value: new value
        """
        self.setValue("model", value)

    @property
    def useMobileNet(self) -> Optional[bool]:
        """
        Getter for useMobileNet

        Returns:
            useMobileNet
        """
        value = self.getValue("useMobileNet")
        if value is None:
            return None
        return bool(value)

    @useMobileNet.setter
    def useMobileNet(self, value: bool) -> None:
        """
        Setter for useMobileNet
        Args:
            value: new value
        """
        self.setValue("useMobileNet", int(value))

    @property
    def distance(self) -> Optional[Distance]:
        """
        Getter for distance

        Returns:
            distance
        """
        value = self.getValue("distance")
        if value is None:
            return None
        return Distance.getEnum(value)

    @distance.setter
    def distance(self, value: Distance) -> None:
        """
        Setter for distance
        Args:
            value: new value
        """
        self.setValue("distance", value.value)

    @property
    def descriptorCountWarningLevel(self) -> Optional[str]:
        """
        Getter for descriptorCountWarningLevel

        Returns:
            descriptorCountWarningLevel
        """
        return self.getValue("descriptorCountWarningLevel")

    @descriptorCountWarningLevel.setter
    def descriptorCountWarningLevel(self, value: str) -> None:
        """
        Setter for descriptorCountWarningLevel
        Args:
            value: new value
        """
        self.setValue("descriptorCountWarningLevel", value)


class FaceDetV3Settings(BaseSettingsSection):
    """
    FaceDetV3 detector settings.

    Properties:

        - scoreThreshold (float): detection threshold in [0..1] range;
        - redetectScoreThreshold (float): redetect face threshold in [0..1] range;
        - NMSThreshold (float): overlap threshold for NMS [0..1] range;
        - minFaceSize (int): Minimum face size in pixels;
        - maxFaceSize (int): Maximum face size in pixels;
        - nms (NMS): type of NMS: mean or best;
        - redetectTensorSize (int): target face after preprocessing for redetect;
        - redetectFaceTargetSize (int): target face size for redetect;
        - paddings (Point4): paddings;
        - paddingsIR (Point4): paddingsIR;
        - planPrefix (str): planPrefix;
        - useOrientationMode (bool): use mode for rotated origin images or not;
    """

    sectionName = "FaceDetV3::Settings"

    @property
    def scoreThreshold(self) -> Optional[float]:
        """
        Getter for scoreThreshold

        Returns:
            scoreThreshold
        """
        return self.getValue("scoreThreshold")

    @scoreThreshold.setter
    def scoreThreshold(self, value: float) -> None:
        """
        Setter for descriptorCountWarningLevel
        Args:
            value: new value
        """
        self.setValue("ScoreThreshold", value)

    @property
    def redetectScoreThreshold(self) -> Optional[float]:
        """
        Getter for redetectScoreThreshold

        Returns:
            redetectScoreThreshold
        """
        return self.getValue("RedetectScoreThreshold")

    @redetectScoreThreshold.setter
    def redetectScoreThreshold(self, value: float) -> None:
        """
        Setter for redetectScoreThreshold
        Args:
            value: new value
        """
        self.setValue("RedetectScoreThreshold", value)

    @property
    def NMSThreshold(self) -> Optional[float]:
        """
        Getter for NMSThreshold

        Returns:
            NMSThreshold
        """
        return self.getValue("NMSThreshold")

    @NMSThreshold.setter
    def NMSThreshold(self, value: float) -> None:
        """
        Setter for NMSThreshold
        Args:
            value: new value
        """
        self.setValue("NMSThreshold", value)

    @property
    def minFaceSize(self) -> Optional[int]:
        """
        Getter for minFaceSize

        Returns:
            minFaceSize
        """
        return self.getValue("minFaceSize")

    @minFaceSize.setter
    def minFaceSize(self, value: int) -> None:
        """
        Setter for minFaceSize
        Args:
            value: new value
        """
        self.setValue("minFaceSize", value)

    @property
    def maxFaceSize(self) -> Optional[int]:
        """
        Getter for maxFaceSize

        Returns:
            maxFaceSize
        """
        return self.getValue("maxFaceSize")

    @maxFaceSize.setter
    def maxFaceSize(self, value: int) -> None:
        """
        Setter for maxFaceSize
        Args:
            value: new value
        """
        self.setValue("maxFaceSize", value)

    @property
    def nms(self) -> Optional[NMS]:
        """
        Getter for nms

        Returns:
            nms
        """
        value = self.getValue("nms")
        if value is None:
            return None
        return NMS[value]

    @nms.setter
    def nms(self, value: NMS) -> None:
        """
        Setter for nms
        Args:
            value: new value
        """
        self.setValue("nms", value.value)

    @property
    def redetectTensorSize(self) -> Optional[int]:
        """
        Getter for redetectTensorSize

        Returns:
            redetectTensorSize
        """
        return self.getValue("RedetectTensorSize")

    @redetectTensorSize.setter
    def redetectTensorSize(self, value: int) -> None:
        """
        Setter for redetectTensorSize
        Args:
            value: new value
        """
        self.setValue("RedetectTensorSize", value)

    @property
    def redetectFaceTargetSize(self) -> Optional[int]:
        """
        Getter for redetectFaceTargetSize

        Returns:
            redetectFaceTargetSize
        """
        return self.getValue("RedetectFaceTargetSize")

    @redetectFaceTargetSize.setter
    def redetectFaceTargetSize(self, value: int) -> None:
        """
        Setter for redetectFaceTargetSize
        Args:
            value: new value
        """
        self.setValue("RedetectFaceTargetSize", value)

    @property
    def paddings(self) -> Optional[Point4]:
        """
        Getter for paddings

        Returns:
            paddings
        """
        value = self.getValue("paddings")
        if value is None:
            return None
        return Point4(*value)

    @paddings.setter
    def paddings(self, value: Point4) -> None:
        """
        Setter for paddings
        Args:
            value: new value
        """
        self.setValue("paddings", value.asTuple())

    @property
    def paddingsIR(self) -> Optional[Point4]:
        """
        Getter for paddingsIR

        Returns:
            paddingsIR
        """
        value = self.getValue("paddingsIR")
        if value is None:
            return None
        return Point4(*value)

    @paddingsIR.setter
    def paddingsIR(self, value: Point4) -> None:
        """
        Setter for paddingsIR
        Args:
            value: new value
        """
        self.setValue("paddingsIR", value.asTuple())

    @property
    def planPrefix(self) -> Optional[str]:
        """
        Getter for planPrefix

        Returns:
            planPrefix
        """
        return self.getValue("planPrefix")

    @planPrefix.setter
    def planPrefix(self, value: str) -> None:
        """
        Setter for planPrefix
        Args:
            value: new value
        """
        self.setValue("planPrefix", value)

    @property
    def useOrientationMode(self) -> Optional[bool]:
        """
        Getter for useOrientationMode

        Returns:
            useEstimationByImage
        """
        value = self.getValue("useOrientationM4de")
        if value is None:
            return None
        return bool(value)

    @useOrientationMode.setter
    def useOrientationMode(self, value: bool) -> None:
        """
        Setter for useOrientationMode
        Args:
            value: new value
        """
        self.setValue("useOrientationMode", int(value))


class FaceDetV12Settings(BaseSettingsSection):
    """
    Common class for FaceDetV1 and FaceDetV2 detector settings.

    Properties:

        - firstThreshold (float): 1-st threshold in [0..1] range;
        - secondThreshold (float): 2-st threshold in [0..1] range;
        - thirdTThreshold (float): 3-st threshold in [0..1] range;
        - minFaceSize (int): minimum face size in pixels;
        - scaleFactor (float): image scale factor;
        - paddings (Point4): paddings;
        - redetectTolerance (float): redetect tolerance;
    """

    @property
    def firstThreshold(self) -> Optional[float]:
        """
        Getter for firstThreshold

        Returns:
            float in [0..1] range
        """
        return self.getValue("FirstThreshold")

    @firstThreshold.setter
    def firstThreshold(self, value: float) -> None:
        """
        Setter for firstThreshold
        Args:
            value: new value, float in [0..1] range
        """
        self.setValue("FirstThreshold", value)

    @property
    def secondThreshold(self) -> Optional[float]:
        """
        Getter for secondThreshold

        Returns:
            secondThreshold
        """
        return self.getValue("SecondThreshold")

    @secondThreshold.setter
    def secondThreshold(self, value: float) -> None:
        """
        Setter for secondThreshold
        Args:
            value: new value, float in [0..1] range
        """
        self.setValue("SecondThreshold", value)

    @property
    def thirdThreshold(self) -> Optional[float]:
        """
        Getter for thirdThreshold

        Returns:
            thirdThreshold
        """
        return self.getValue("ThirdThreshold")

    @thirdThreshold.setter
    def thirdThreshold(self, value: float) -> None:
        """
        Setter for thirdThreshold
        Args:
            value: new value, float in [0..1] range
        """
        self.setValue("ThirdThreshold", value)

    @property
    def minFaceSize(self) -> Optional[int]:
        """
        Getter for minFaceSize

        Returns:
            minSize
        """
        return self.getValue("minFaceSize")

    @minFaceSize.setter
    def minFaceSize(self, value: int) -> None:
        """
        Setter for minFaceSize
        Args:
            value: new value
        """
        self.setValue("minFaceSize", value)

    @property
    def scaleFactor(self) -> Optional[float]:
        """
        Getter for scaleFactor

        Returns:
            scaleFactor
        """
        return self.getValue("scaleFactor")

    @scaleFactor.setter
    def scaleFactor(self, value: float) -> None:
        """
        Setter for descriptorCountWarningLevel
        Args:
            value: new value
        """
        self.setValue("scaleFactor", value)

    @property
    def paddings(self) -> Optional[Point4]:
        """
        Getter for paddings

        Returns:
            paddings
        """
        value = self.getValue("paddings")
        if value is None:
            return None
        return Point4(*value)

    @paddings.setter
    def paddings(self, value: Point4) -> None:
        """
        Setter for descriptorCountWarningLevel
        Args:
            value: new value
        """
        self.setValue("paddings", value.asTuple())

    @property
    def redetectTolerance(self) -> Optional[float]:
        """
        Getter for redetectTolerance

        Returns:
            redetectTolerance
        """
        return self.getValue("redetectTolerance")

    @redetectTolerance.setter
    def redetectTolerance(self, value: float) -> None:
        """
        Setter for descriptorCountWarningLevel
        Args:
            value: new value
        """
        self.setValue("redetectTolerance", value)

    @property
    def useLNet(self) -> Optional[bool]:
        """
        Getter for useLNet

        Returns:
            useEstimationByImage
        """
        value = self.getValue("useLNet")
        if value is None:
            return None
        return bool(value)

    @useLNet.setter
    def useLNet(self, value: bool) -> None:
        """
        Setter for useLNet
        Args:
            value: new value
        """
        self.setValue("useLNet", int(value))


class FaceDetV1Settings(FaceDetV12Settings):
    """
    FaceDetV1 settings.
    """

    sectionName = "FaceDetV1::Settings"


class FaceDetV2Settings(FaceDetV12Settings):
    """
    FaceDetV2 settings.
    """

    sectionName = "FaceDetV2::Settings"


class BodyDetectorSettings(BaseSettingsSection):
    """
    BodyDetector detector settings.

    Properties:

        - scoreThreshold (float): detection threshold in [0..1] range;
        - NMSThreshold (float): overlap threshold for NMS [0..1] range;
        - imageSize (int): Target image size for down scaling by load side;
        - nms (NMS): type of NMS: mean or best
        - redetectNMS: type of NMS: mean or best
        - landmarks17Threshold (float): body landmarks threshold in [0..1] range;
    """

    sectionName = "HumanDetector::Settings"

    @property
    def scoreThreshold(self) -> Optional[float]:
        """
        Getter for scoreThreshold

        Returns:
            scoreThreshold
        """
        return self.getValue("scoreThreshold")

    @scoreThreshold.setter
    def scoreThreshold(self, value: float) -> None:
        """
        Setter for descriptorCountWarningLevel
        Args:
            value: new value
        """
        self.setValue("ScoreThreshold", value)

    @property
    def redetectScoreThreshold(self) -> Optional[float]:
        """
        Getter for redetectScoreThreshold

        Returns:
            redetectScoreThreshold
        """
        return self.getValue("RedetectScoreThreshold")

    @redetectScoreThreshold.setter
    def redetectScoreThreshold(self, value: float) -> None:
        """
        Setter for redetectScoreThreshold
        Args:
            value: new value
        """
        self.setValue("RedetectScoreThreshold", value)

    @property
    def NMSThreshold(self) -> Optional[float]:
        """
        Getter for NMSThreshold

        Returns:
            NMSThreshold
        """
        return self.getValue("NMSThreshold")

    @NMSThreshold.setter
    def NMSThreshold(self, value: float) -> None:
        """
        Setter for NMSThreshold
        Args:
            value: new value
        """
        self.setValue("NMSThreshold", value)

    @property
    def redetectNMSThreshold(self) -> Optional[float]:
        """
        Getter for redetectNMSThreshold

        Returns:
            redetectNMSThreshold
        """
        return self.getValue("RedetectNMSThreshold")

    @redetectNMSThreshold.setter
    def redetectNMSThreshold(self, value: float) -> None:
        """
        Setter for redetectNMSThreshold
        Args:
            value: new value
        """
        self.setValue("RedetectNMSThreshold", value)

    @property
    def imageSize(self) -> Optional[int]:
        """
        Getter for imageSize

        Returns:
            imageSize
        """
        return self.getValue("imageSize")

    @imageSize.setter
    def imageSize(self, value: int) -> None:
        """
        Setter for imageSize
        Args:
            value: new value
        """
        self.setValue("imageSize", value)

    @property
    def nms(self) -> Optional[NMS]:
        """
        Getter for nms

        Returns:
            nms
        """
        value = self.getValue("nms")
        if value is None:
            return None
        return NMS[value]

    @nms.setter
    def nms(self, value: NMS) -> None:
        """
        Setter for nms
        Args:
            value: new value
        """
        self.setValue("nms", value.value)

    @property
    def redetectNMS(self) -> Optional[NMS]:
        """
        Getter for redetectMms

        Returns:
            nms
        """
        value = self.getValue("RedetectNMS")
        if value is None:
            return None
        return NMS[value]

    @redetectNMS.setter
    def redetectNMS(self, value: NMS) -> None:
        """
        Setter for redetectNMS
        Args:
            value: redetectNMS value
        """
        self.setValue("RedetectNMS", value.value)

    @property
    def landmarks17Threshold(self) -> Optional[float]:
        """
        Getter for landmarks17Threshold

        Returns:
            scoreThreshold
        """
        return self.getValue("landmarks17Threshold")

    @landmarks17Threshold.setter
    def landmarks17Threshold(self, value: float) -> None:
        """
        Setter for landmarks17Threshold
        Args:
            value: new value
        """
        self.setValue("landmarks17Threshold", value)


class HumanDetectorSettings(BaseSettingsSection):
    """
    HumanDetector detector settings.

    Properties:

        - faceThreshold (float): face detection threshold in [0..1] range;
        - bodyThreshold (float): face detection threshold in [0..1] range;
        - associationThreshold (float): body and face association threshold in [0..1] range;
        - minFaceSize (int): Minimum face size in pixels;
        - nmsFaceThreshold (float): overlap threshold for face NMS [0..1] range;
        - nmsBodyThreshold (float): overlap threshold for face NMS [0..1] range;
    """

    sectionName = "HumanFaceDetector::Settings"

    @property
    def faceThreshold(self) -> Optional[float]:
        """
        Getter for faceThreshold

        Returns:
            faceThreshold
        """
        return self.getValue("faceThreshold")

    @faceThreshold.setter
    def faceThreshold(self, value: float) -> None:
        """
        Setter for faceThreshold
        Args:
            value: new value
        """
        self.setValue("faceThreshold", value)

    @property
    def bodyThreshold(self) -> Optional[float]:
        """
        Getter for humanThreshold

        Returns:
            scoreThreshold
        """
        return self.getValue("humanThreshold")

    @bodyThreshold.setter
    def bodyThreshold(self, value: float) -> None:
        """
        Setter for humanThreshold
        Args:
            value: new value
        """
        self.setValue("humanThreshold", value)

    @property
    def associationThreshold(self) -> Optional[float]:
        """
        Getter for associationThreshold

        Returns:
            scoreThreshold
        """
        return self.getValue("associationThreshold")

    @associationThreshold.setter
    def associationThreshold(self, value: float) -> None:
        """
        Setter for associationThreshold
        Args:
            value: new value
        """
        self.setValue("associationThreshold", value)

    @property
    def minFaceSize(self) -> Optional[int]:
        """
        Getter for minFaceSize

        Returns:
            minFaceSize
        """
        return self.getValue("minFaceSize")

    @minFaceSize.setter
    def minFaceSize(self, value: int) -> None:
        """
        Setter for minFaceSize
        Args:
            value: new value
        """
        self.setValue("minFaceSize", value)

    @property
    def nmsFaceThreshold(self) -> Optional[float]:
        """
        Getter for nmsFaceThreshold

        Returns:
            nmsFaceThreshold value
        """
        return self.getValue("nmsFaceThreshold")

    @nmsFaceThreshold.setter
    def nmsFaceThreshold(self, value: float) -> None:
        """
        Setter for nmsFaceThreshold
        Args:
            value: new value
        """
        self.setValue("nmsFaceThreshold", value)

    @property
    def nmsBodyThreshold(self) -> Optional[float]:
        """
        Getter for nmsHumanThreshold

        Returns:
            nmsBodyThreshold value
        """
        return self.getValue("nmsHumanThreshold")

    @nmsBodyThreshold.setter
    def nmsBodyThreshold(self, value: float) -> None:
        """
        Setter for nmsHumanThreshold
        Args:
            value: new value
        """
        self.setValue("nmsHumanThreshold", value)


class LNetBaseSettings(BaseSettingsSection):
    """
    Base class for configuration LNet neural network.

    Properties:

        - planName (str): plan name
        - size (int): size
        - mean (Point3): mean
        - sigma (Point3): sigma

    """

    @property
    def planName(self) -> Optional[str]:
        """
        Getter for planName

        Returns:
            planName
        """
        return self.getValue("planName")

    @planName.setter
    def planName(self, value: str) -> None:
        """
        Setter for planName
        Args:
            value: new value
        """
        self.setValue("planName", value)

    @property
    def size(self) -> Optional[int]:
        """
        Getter for size

        Returns:
            size
        """
        return self.getValue("size")

    @size.setter
    def size(self, value: int) -> None:
        """
        Setter for size
        Args:
            value: new value
        """
        self.setValue("size", value)

    @property
    def mean(self) -> Optional[Point3]:
        """
        Getter for mean

        Returns:
            mean
        """
        value = self.getValue("mean")
        if value is None:
            return value
        return Point3(*value)

    @mean.setter
    def mean(self, value: Point3) -> None:
        """
        Setter for mean
        Args:
            value: new value
        """
        self.setValue("mean", value.asTuple())

    @property
    def sigma(self) -> Optional[Point3]:
        """
        Getter for sigma

        Returns:
            sigma
        """
        value = self.getValue("sigma")
        if value is None:
            return None
        return Point3(*value)

    @sigma.setter
    def sigma(self, value: Point3) -> None:
        """
        Setter for sigma
        Args:
            value: new value
        """
        self.setValue("sigma", value.asTuple())


class LNetSettings(LNetBaseSettings):
    """LNet configuration section"""

    sectionName = "LNet::Settings"


class LNetIRSettings(LNetBaseSettings):
    """LNetIR configuration section"""

    sectionName = "LNetIR::Settings"


class SLNetSettings(LNetBaseSettings):
    """SLNet configuration section"""

    sectionName = "SLNet::Settings"


class QualityEstimatorSettings(BaseSettingsSection):
    """
    Quality estimator settings section.

    Properties:

        - size (int): size
        - expLight (Point3): expLight
        - expDark (Point3): expDark
        - expBlur (Point3):  expBlur
        - logGray (Point4): logGray
        - platt (Point2): coefficient platt
    """

    sectionName = "QualityEstimator::Settings"

    @property
    def size(self) -> Optional[int]:
        """
        Getter for size

        Returns:
            size
        """
        return self.getValue("size")

    @size.setter
    def size(self, value: int) -> None:
        """
        Setter for size
        Args:
            value: new value
        """
        self.setValue("size", value)

    @property
    def expLight(self) -> Optional[Point3]:
        """
        Getter for expLight

        Returns:
            expLight
        """
        value = self.getValue("expLight")
        if value is None:
            return None
        return Point3(*value)

    @expLight.setter
    def expLight(self, value: Point3) -> None:
        """
        Setter for expLight
        Args:
            value: new value
        """
        self.setValue("expLight", value.asTuple())

    @property
    def expDark(self) -> Optional[Point3]:
        """
        Getter for expDark

        Returns:
            expDark
        """
        value = self.getValue("expDark")
        if value is None:
            return None
        return Point3(*value)

    @expDark.setter
    def expDark(self, value: Point3) -> None:
        """
        Setter for descriptorCountWarningLevel
        Args:
            value: new expDark
        """
        self.setValue("expDark", value.asTuple())

    @property
    def logGray(self) -> Optional[Point4]:
        """
        Getter for logGray

        Returns:
            logGray
        """
        value = self.getValue("logGray")
        if value is None:
            return None
        return Point4(*value)

    @logGray.setter
    def logGray(self, value: Point4) -> None:
        """
        Setter for logGray
        Args:
            value: new value
        """
        self.setValue("logGray", value.asTuple())

    @property
    def expBlur(self) -> Optional[Point3]:
        """
        Getter for expBlur

        Returns:
            expBlur
        """
        value = self.getValue("expBlur")
        if value is None:
            return None
        return Point3(*value)

    @expBlur.setter
    def expBlur(self, value: Point3) -> None:
        """
        Setter for expBlur
        Args:
            value: new value
        """
        self.setValue("expBlur", value.asTuple())

    @property
    def platt(self) -> Optional[Point2]:
        """
        Getter for platt

        Returns:
            platt
        """
        value = self.getValue("platt")
        if value is None:
            return None
        return Point2(*value)

    @platt.setter
    def platt(self, value: Point2) -> None:
        """
        Setter for platt
        Args:
            value: new value
        """
        self.setValue("platt", value.asTuple())


class HeadPoseEstimatorSettings(BaseSettingsSection):
    """
    Head pose estimator settings section.

    Properties:
        - useEstimationByImage (bool): use head pose estimation by image
        - useEstimationByLandmarks (bool): use head pose estimation by landmarks
    """

    sectionName = "HeadPoseEstimator::Settings"

    @property
    def useEstimationByImage(self) -> Optional[bool]:
        """
        Getter for useEstimationByImage

        Returns:
            useEstimationByImage
        """
        value = self.getValue("useEstimationByImage")
        if value is None:
            return None
        return bool(value)

    @useEstimationByImage.setter
    def useEstimationByImage(self, value: bool) -> None:
        """
        Setter for useEstimationByImage
        Args:
            value: new value
        """
        self.setValue("useEstimationByImage", int(value))

    @property
    def useEstimationByLandmarks(self) -> Optional[bool]:
        """
        Getter for useEstimationByLandmarks

        Returns:
            useEstimationByLandmarks
        """
        value = self.getValue("useEstimationByLandmarks")
        if value is None:
            return None
        return bool(value)

    @useEstimationByLandmarks.setter
    def useEstimationByLandmarks(self, value: bool) -> None:
        """
        Setter for useEstimationByLandmarks
        Args:
            value: new value
        """
        self.setValue("useEstimationByLandmarks", int(value))


class EyeEstimatorSettings(BaseSettingsSection):
    """
    Eyes estimator settings section.

    Properties:
        - useStatusPlan (bool): use  status plan or not.
    """

    sectionName = "EyeEstimator::Settings"

    @property
    def useStatusPlan(self) -> Optional[bool]:
        """
        Getter for useStatusPlan

        Returns:
            useStatusPlan
        """
        value = self.getValue("useStatusPlan")
        if value is None:
            return None
        return bool(value)

    @useStatusPlan.setter
    def useStatusPlan(self, value: bool) -> None:
        """
        Setter for useStatusPlan
        Args:
            value: new value
        """
        self.setValue("useStatusPlan", int(value))


class BestShotQualityEstimatorSettings(BaseSettingsSection):
    """
    Best shot quality estimator estimator settings section.

    Properties:
        - runSubestimatorsConcurrently (int): run sub estimators concurrently
    """

    sectionName = "BestShotQualityEstimator::Settings"

    @property
    def runSubestimatorsConcurrently(self) -> Optional[int]:
        """
        Getter for runSubestimatorsConcurrently

        Returns:
            useStatusPlan
        """
        return self.getValue("runSubestimatorsConcurrently")

    @runSubestimatorsConcurrently.setter
    def runSubestimatorsConcurrently(self, value: int) -> None:
        """
        Setter for runSubestimatorsConcurrently
        Args:
            value: new value
        """
        self.setValue("runSubestimatorsConcurrently", value)


class AttributeEstimatorSettings(BaseSettingsSection):
    """
    Attribute estimator settings section.

    Properties:
        - genderThreshold (float): gender threshold in [0..1] range
        - adultThreshold (float): adult threshold in [0..1] range
    """

    sectionName = "AttributeEstimator::Settings"

    @property
    def genderThreshold(self) -> Optional[float]:
        """
        Getter for genderThreshold

        Returns:
            genderThreshold
        """
        return self.getValue("genderThreshold")

    @genderThreshold.setter
    def genderThreshold(self, value: float) -> None:
        """
        Setter for genderThreshold
        Args:
            value: new value
        """
        self.setValue("genderThreshold", value)

    @property
    def adultThreshold(self) -> Optional[float]:
        """
        Getter for adultThreshold

        Returns:
            adultThreshold
        """
        return self.getValue("adultThreshold")

    @adultThreshold.setter
    def adultThreshold(self, value: float) -> None:
        """
        Setter for adultThreshold
        Args:
            value: new value
        """
        self.setValue("adultThreshold", value)


class GlassesEstimatorSettings(BaseSettingsSection):
    """
    Glasses estimator settings section.

    Properties:
        - noGlassesThreshold (float): no glasses threshold in [0..1] range
        - eyeGlassesThreshold (float): eye glasses threshold in [0..1] range
        - sunGlassesThreshold (float): sun glasses threshold in [0..1] range
    """

    sectionName = "GlassesEstimator::Settings"

    @property
    def noGlassesThreshold(self) -> Optional[float]:
        """
        Getter for noGlassesThreshold

        Returns:
            noGlassesThreshold
        """
        return self.getValue("noGlassesThreshold")

    @noGlassesThreshold.setter
    def noGlassesThreshold(self, value: float) -> None:
        """
        Setter for noGlassesThreshold
        Args:
            value: new value
        """
        self.setValue("noGlassesThreshold", value)

    @property
    def eyeGlassesThreshold(self) -> Optional[float]:
        """
        Getter for eyeGlassesThreshold

        Returns:
            eyeGlassesThreshold
        """
        return self.getValue("eyeGlassesThreshold")

    @eyeGlassesThreshold.setter
    def eyeGlassesThreshold(self, value: float) -> None:
        """
        Setter for eyeGlassesThreshold
        Args:
            value: new value
        """
        self.setValue("eyeGlassesThreshold", value)

    @property
    def sunGlassesThreshold(self) -> Optional[float]:
        """
        Getter for sunGlassesThreshold

        Returns:
            sunGlassesThreshold
        """
        return self.getValue("sunGlassesThreshold")

    @sunGlassesThreshold.setter
    def sunGlassesThreshold(self, value: float) -> None:
        """
        Setter for sunGlassesThreshold
        Args:
            value: new value
        """
        self.setValue("sunGlassesThreshold", value)


class OverlapEstimatorSettings(BaseSettingsSection):
    """
    OverlapThreshold any object settings section.

    Properties:
        - overlapThreshold (float): overlap threshold for any object in [0..1] range
    """

    sectionName = "OverlapEstimator::Settings"

    @property
    def overlapThreshold(self) -> Optional[float]:
        """
        Getter for overlapThreshold

        Returns:
            overlapThreshold
        """
        return self.getValue("overlapThreshold")

    @overlapThreshold.setter
    def overlapThreshold(self, value: float) -> None:
        """
        Setter for overlapThreshold
        Args:
            value: new value
        """
        self.setValue("overlapThreshold", value)


class ChildEstimatorSettings(BaseSettingsSection):
    """
    childThreshold settings section.

    Properties:
        - childThreshold (float):  if estimate value less than threshold object is a children.
    """

    sectionName = "ChildEstimator::Settings"

    @property
    def childThreshold(self) -> Optional[float]:
        """
        Getter for childThreshold

        Returns:
            childThreshold
        """
        return self.getValue("childThreshold")

    @childThreshold.setter
    def childThreshold(self, value: float) -> None:
        """
        Setter for childThreshold
        Args:
            value: new value
        """
        self.setValue("childThreshold", value)


class LivenessIREstimatorSettings(BaseSettingsSection):
    """
    LivenessIREstimator settings section.

    Properties:
        - cooperativeMode (bool): whether liveness is checking in cooperative mode
        - irCooperativeThreshold (float): liveness threshold for cooperative mode in [0..1] range
        - irNonCooperativeThreshold (float): liveness threshold for non cooperative mode in [0..1] range
    """

    sectionName = "LivenessIREstimator::Settings"

    @property
    def cooperativeMode(self) -> Optional[bool]:
        """
        Getter for cooperativeMode

        Returns:
            cooperativeMode
        """
        value = self.getValue("cooperativeMode")
        if value is None:
            return None
        return bool(value)

    @cooperativeMode.setter
    def cooperativeMode(self, value: bool) -> None:
        """
        Setter for cooperativeMode
        Args:
            value: new value
        """
        self.setValue("cooperativeMode", int(value))

    @property
    def irCooperativeThreshold(self) -> Optional[float]:
        """
        Getter for irCooperativeThreshold

        Returns:
            irCooperativeThreshold
        """
        return self.getValue("irCooperativeThreshold")

    @irCooperativeThreshold.setter
    def irCooperativeThreshold(self, value: float) -> None:
        """
        Setter for irCooperativeThreshold
        Args:
            value: new value
        """
        self.setValue("irCooperativeThreshold", value)

    @property
    def irNonCooperativeThreshold(self) -> Optional[float]:
        """
        Getter for irNonCooperativeThreshold

        Returns:
            irNonCooperativeThreshold
        """
        return self.getValue("irNonCooperativeThreshold")

    @irNonCooperativeThreshold.setter
    def irNonCooperativeThreshold(self, value: float) -> None:
        """
        Setter for irNonCooperativeThreshold
        Args:
            value: new value
        """
        self.setValue("irNonCooperativeThreshold", value)


class MaskEstimatorSettings(BaseSettingsSection):
    """
    MaskEstimatorSettings settings section.

    Properties:
        - medicalMaskThreshold (float): medical mask state threshold in [0..1] range
        - missingThreshold (float): missing mask state threshold in [0..1] range
        - occludedThreshold (float): occluded mask state threshold in [0..1] range
    """

    sectionName = "MedicalMaskEstimator::Settings"

    @property
    def medicalMaskThreshold(self) -> Optional[float]:
        """
        Getter for medicalMaskThreshold

        Returns:
            medicalMaskThreshold
        """
        return self.getValue("maskThreshold")

    @medicalMaskThreshold.setter
    def medicalMaskThreshold(self, value: float) -> None:
        """
        Setter for medicalMaskThreshold
        Args:
            value: new value
        """
        self.setValue("maskThreshold", value)

    @property
    def missingThreshold(self) -> Optional[float]:
        """
        Getter for missingThreshold

        Returns:
            missingThreshold
        """
        return self.getValue("noMaskThreshold")

    @missingThreshold.setter
    def missingThreshold(self, value: float) -> None:
        """
        Setter for missingThreshold
        Args:
            value: new value
        """
        self.setValue("noMaskThreshold", value)

    @property
    def occludedThreshold(self) -> Optional[float]:
        """
        Getter for occludedThreshold

        Returns:
            occludedThreshold
        """
        return self.getValue("occludedFaceThreshold")

    @occludedThreshold.setter
    def occludedThreshold(self, value: float) -> None:
        """
        Setter for occludedThreshold
        Args:
            value: new value
        """
        self.setValue("occludedFaceThreshold", value)


class MouthEstimatorSettings(BaseSettingsSection):
    """
    MouthEstimator settings section.

    Properties:
        - occlusionThreshold (float): occlusion mouth threshold in [0..1] range
        - smileThreshold (float): smile threshold in [0..1] range
        - openThreshold (float): open mouth threshold in [0..1] range
    """

    sectionName = "MouthEstimator::Settings"

    @property
    def occlusionThreshold(self) -> Optional[float]:
        """
        Getter for occlusionThreshold

        Returns:
            occlusionThreshold
        """
        return self.getValue("occlusionThreshold")

    @occlusionThreshold.setter
    def occlusionThreshold(self, value: float) -> None:
        """
        Setter for medicalMaskThreshold
        Args:
            value: new value
        """
        self.setValue("occlusionThreshold", value)

    @property
    def smileThreshold(self) -> Optional[float]:
        """
        Getter for smileThreshold

        Returns:
            smileThreshold
        """
        return self.getValue("smileThreshold")

    @smileThreshold.setter
    def smileThreshold(self, value: float) -> None:
        """
        Setter for smileThreshold
        Args:
            value: new value
        """
        self.setValue("smileThreshold", value)

    @property
    def openThreshold(self) -> Optional[float]:
        """
        Getter for openThreshold

        Returns:
            openThreshold
        """
        return self.getValue("openThreshold")

    @openThreshold.setter
    def openThreshold(self, value: float) -> None:
        """
        Setter for occludedThreshold
        Args:
            value: new value
        """
        self.setValue("openThreshold", value)


class HeadAndShouldersLivenessEstimatorSettings(BaseSettingsSection):
    """
    HeadAndShouldersLiveness settings section.

    Properties:
        - shouldersHeightKoeff (float): shouldersHeightKoeff
        - shouldersWidthKoeff (float): shouldersWidthKoeff
        - headWidthKoeff (float): headWidthKoeff
        - headHeightKoeff (float): headHeightKoeff
    """

    sectionName = "HeadAndShouldersLivenessEstimator::Settings"

    @property
    def headWidthKoeff(self) -> Optional[float]:
        """
        Getter for betaMode

        Returns:
            betaMode
        """
        return self.getValue("headWidthKoeff")

    @headWidthKoeff.setter
    def headWidthKoeff(self, value: float) -> None:
        """
        Setter for descriptorCountWarningLevel
        Args:
            value: new value
        """
        self.setValue("headWidthKoeff", value)

    @property
    def headHeightKoeff(self) -> Optional[float]:
        """
        Getter for betaMode

        Returns:
            betaMode
        """
        return self.getValue("headHeightKoeff")

    @headHeightKoeff.setter
    def headHeightKoeff(self, value: float) -> None:
        """
        Setter for descriptorCountWarningLevel
        Args:
            value: new value
        """
        self.setValue("headHeightKoeff", value)

    @property
    def shouldersWidthKoeff(self) -> Optional[float]:
        """
        Getter for betaMode

        Returns:
            betaMode
        """
        return self.getValue("shouldersWidthKoeff")

    @shouldersWidthKoeff.setter
    def shouldersWidthKoeff(self, value: float) -> None:
        """
        Setter for descriptorCountWarningLevel
        Args:
            value: new value
        """
        self.setValue("shouldersWidthKoeff", value)

    @property
    def shouldersHeightKoeff(self) -> Optional[float]:
        """
        Getter for betaMode

        Returns:
            betaMode
        """
        return self.getValue("shouldersHeightKoeff")

    @shouldersHeightKoeff.setter
    def shouldersHeightKoeff(self, value: float) -> None:
        """
        Setter for descriptorCountWarningLevel
        Args:
            value: new value
        """
        self.setValue("shouldersHeightKoeff", value)


class LivenessV1Estimator(BaseSettingsSection):
    """
    LivenessV1(LivenessOneShotRGBEstimator) settings section.

    Properties:
        - realThreshold (float): realThreshold
        - useFilter (bool): useFilter
        - minDetSize (int): minDetSize
        - borderDistance (int): borderDistance
        - principalAxes(int): principalAxes
    """

    sectionName = "LivenessOneShotRGBEstimator::Settings"

    @property
    def realThreshold(self) -> Optional[float]:
        """
        Getter for realThreshold

        Returns:
            realThreshold
        """
        return self.getValue("realThreshold")

    @realThreshold.setter
    def realThreshold(self, value: float) -> None:
        """
        Setter for realThreshold
        Args:
            value: new value
        """
        self.setValue("realThreshold", value)

    @property
    def qualityThreshold(self) -> Optional[float]:
        """
        Getter for qualityThreshold

        Returns:
            realThreshold
        """
        return self.getValue("qualityThreshold")

    @qualityThreshold.setter
    def qualityThreshold(self, value: float) -> None:
        """
        Setter for qualityThreshold
        Args:
            value: new value
        """
        self.setValue("qualityThreshold", value)


class BaseSettingsProvider:
    """
    Runtime SDK Setting faceEngineProvider.

    Proxy model.

    Attributes:
        pathToConfig (str): path to a configuration file. Config file is getting from
                          the folder'data'  in "FSDK_ROOT".
        _coreSettingProvider (PyISettingsProvider): core settings provider
    """

    # default configuration filename.
    defaultConfName = ""

    def __init__(self, pathToConfig: Optional[Union[str, Path]] = None):
        """
        Init.

        Args:
            pathToConfig: path to config.
        Raises:
             ValueError: if pathToConfig is None and environment variable *FSDK_ROOT* does not set.
        """
        if pathToConfig is None:
            if "FSDK_ROOT" in os.environ:
                self.pathToConfig = Path(os.environ["FSDK_ROOT"]).joinpath("data", self.defaultConfName)
            else:
                raise ValueError(
                    "Failed on path to faceengine luna data folder, set variable pathToData or set"
                    "environment variable *FSDK_ROOT*"
                )
        elif isinstance(pathToConfig, str):
            self.pathToConfig = Path(pathToConfig)
        else:
            self.pathToConfig = pathToConfig

        # todo: check existance

        self._coreSettingProvider = CoreFE.createSettingsProvider(str(self.pathToConfig))

    @property
    def coreProvider(self) -> PyISettingsProvider:
        """
        Get core settings provider
        Returns:
            _coreSettingProvider
        """
        return self._coreSettingProvider


class FaceEngineSettingsProvider(BaseSettingsProvider):
    """
    SDK Setting faceEngineProvider.

    Proxy model.
    """

    # default configuration filename.
    defaultConfName = "faceengine.conf"

    @property
    def systemSettings(self) -> SystemSettings:
        """
        Getter for system settings section.

        Returns:
            Mutable system section
        """
        return SystemSettings(self._coreSettingProvider)

    @property
    def descriptorFactorySettings(self) -> DescriptorFactorySettings:
        """
        Getter for descriptor factory settings section.

        Returns:
            Mutable descriptor factory section
        """
        return DescriptorFactorySettings(self._coreSettingProvider)

    @property
    def faceDetV3Settings(self) -> FaceDetV3Settings:
        """
        Getter for FaceDetV3 settings section.

        Returns:
            Mutable FaceDetV3 section
        """
        return FaceDetV3Settings(self._coreSettingProvider)

    @property
    def faceDetV1Settings(self) -> FaceDetV1Settings:
        """
        Getter for FaceDetV1 settings section.

        Returns:
            Mutable FaceDetV1 section
        """
        return FaceDetV1Settings(self._coreSettingProvider)

    @property
    def faceDetV2Settings(self) -> FaceDetV2Settings:
        """
        Getter for FaceDetV2 settings section.

        Returns:
            Mutable FaceDetV2 section
        """
        return FaceDetV2Settings(self._coreSettingProvider)

    @property
    def bodyDetectorSettings(self) -> BodyDetectorSettings:
        """
        Getter for human body settings section.

        Returns:
            Mutable HumanDetectorSettings section
        """
        return BodyDetectorSettings(self._coreSettingProvider)

    @property
    def lNetSettings(self) -> LNetSettings:
        """
        Getter for LNet settings section.

        Returns:
            Mutable LNet section
        """
        return LNetSettings(self._coreSettingProvider)

    @property
    def lNetIRSettings(self) -> LNetIRSettings:
        """
        Getter for LNetIR settings section.

        Returns:
            Mutable LNetIR section
        """
        return LNetIRSettings(self._coreSettingProvider)

    @property
    def slNetSettings(self) -> SLNetSettings:
        """
        Getter for SLNet settings section.

        Returns:
            Mutable SLNet section
        """
        return SLNetSettings(self._coreSettingProvider)

    @property
    def qualityEstimatorSettings(self) -> QualityEstimatorSettings:
        """
        Getter for QualityEstimator settings section.

        Returns:
            Mutable QualityEstimator section
        """
        return QualityEstimatorSettings(self._coreSettingProvider)

    @property
    def headPoseEstimatorSettings(self) -> HeadPoseEstimatorSettings:
        """
        Getter for HeadPoseEstimator settings section.

        Returns:
            Mutable HeadPoseEstimator section
        """
        return HeadPoseEstimatorSettings(self._coreSettingProvider)

    @property
    def eyeEstimatorSettings(self) -> EyeEstimatorSettings:
        """
        Getter for EyeEstimator settings section.

        Returns:
            Mutable EyeEstimator section
        """
        return EyeEstimatorSettings(self._coreSettingProvider)

    @property
    def attributeEstimatorSettings(self) -> AttributeEstimatorSettings:
        """
        Getter for AttributeEstimator settings section.

        Returns:
            Mutable AttributeEstimator section
        """
        return AttributeEstimatorSettings(self._coreSettingProvider)

    @property
    def glassesEstimatorSettings(self) -> GlassesEstimatorSettings:
        """
        Getter for GlassesEstimator settings section.

        Returns:
            Mutable GlassesEstimator section
        """
        return GlassesEstimatorSettings(self._coreSettingProvider)

    @property
    def overlapEstimatorSettings(self) -> OverlapEstimatorSettings:
        """
        Getter for OverlapEstimator settings section.

        Returns:
            Mutable OverlapEstimator section
        """
        return OverlapEstimatorSettings(self._coreSettingProvider)

    @property
    def childEstimatorSettings(self) -> ChildEstimatorSettings:
        """
        Getter for ChildEstimator settings section.

        Returns:
            Mutable ChildEstimator section
        """
        return ChildEstimatorSettings(self._coreSettingProvider)

    @property
    def livenessIREstimatorSettings(self) -> LivenessIREstimatorSettings:
        """
        Getter for LivenessIREstimator settings section.

        Returns:
            Mutable LivenessIREstimator section
        """
        return LivenessIREstimatorSettings(self._coreSettingProvider)

    @property
    def headAndShouldersLivenessEstimatorSettings(self) -> HeadAndShouldersLivenessEstimatorSettings:
        """
        Getter for HeadAndShouldersLivenessEstimator settings section.

        Returns:
            Mutable HeadAndShouldersLivenessEstimator section
        """
        return HeadAndShouldersLivenessEstimatorSettings(self._coreSettingProvider)

    @property
    def bestShotQualityEstimator(self) -> BestShotQualityEstimatorSettings:
        """
        Getter for BestShotQualityEstimatorSettings settings section.

        Returns:
            Mutable BestShotQualityEstimatorSettings section
        """
        return BestShotQualityEstimatorSettings(self._coreSettingProvider)

    @property
    def livenessV1Estimator(self) -> LivenessV1Estimator:
        """
        Getter for LivenessV1Estimator (LivenessOneShotRGBEstimator) settings section.

        Returns:
            Mutable LivenessV1Estimator section
        """
        return LivenessV1Estimator(self._coreSettingProvider)

    @property
    def humanDetectorSettings(self) -> HumanDetectorSettings:
        """
        Getter for HumanDetectorSettings (HumanFaceDetector settings) settings section.

        Returns:
            Mutable HumanDetectorSettings section
        """
        return HumanDetectorSettings(self._coreSettingProvider)


class RuntimeSettingsProvider(BaseSettingsProvider):
    """
    Runtime SDK Setting faceEngineProvider.

    Proxy model.
    """

    defaultConfName = "runtime.conf"

    @property
    def runtimeSettings(self) -> RuntimeSettings:
        """
        Getter for runtime settings section.

        Returns:
            Mutable runtime section
        """
        return RuntimeSettings(self._coreSettingProvider)
