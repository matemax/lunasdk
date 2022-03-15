"""
High-level api for estimating face attributes
"""
from dataclasses import dataclass
from typing import Optional, Union, List, Dict

from FaceEngine import Face  # pylint: disable=E0611,E0401
from FaceEngine import Image as CoreImage  # pylint: disable=E0611,E0401
from PIL.Image import Image as PilImage
from numpy import ndarray

from .detectors.base import ImageForDetection, ImageForRedetection
from .detectors.facedetector import FaceDetection, FaceDetector, Landmarks5
from .estimator_collections import FaceEstimatorsCollection
from .estimators.base import ImageWithFaceDetection
from .estimators.face_estimators.basic_attributes import BasicAttributes
from .estimators.face_estimators.credibility import Credibility
from .estimators.face_estimators.emotions import Emotions
from .estimators.face_estimators.eyes import EyesEstimation, GazeDirection, WarpWithLandmarks, WarpWithLandmarks5
from .estimators.face_estimators.face_descriptor import FaceDescriptor
from .estimators.face_estimators.facewarper import FaceWarp, FaceWarpedImage
from .estimators.face_estimators.glasses import Glasses
from .estimators.face_estimators.head_pose import HeadPose
from .estimators.face_estimators.livenessv1 import LivenessV1
from .estimators.face_estimators.mask import Mask
from .estimators.face_estimators.mouth_state import MouthStates
from .estimators.face_estimators.warp_quality import Quality
from .faceengine.engine import VLFaceEngine
from .faceengine.setting_provider import DetectorType
from .image_utils.geometry import Rect
from .image_utils.image import VLImage, ColorFormat


@dataclass(frozen=True)
class VLFaceDetectionSettings:
    """
    Settings for detection
    """

    # estimate mask from detection or warp
    estimateMaskFromDetection: bool = False


class VLFaceDetection(FaceDetection):
    """
    High level detection object.
    Attributes:
        _estimationSettings (VLFaceDetectionSettings): settings for detections
        estimatorCollection (FaceEstimatorsCollection): collection of estimators
        _emotions (Optional[Emotions]): lazy load emotions estimations
        _eyes (Optional[EyesEstimation]): lazy load eye estimations
        _mouthState (Optional[MouthStates]): lazy load mouth state estimation
        _basicAttributes (Optional[BasicAttributes]): lazy load basic attribute estimation
        _gaze (Optional[GazeEstimation]): lazy load gaze direction estimation
        _warpQuality (Optional[Quality]): lazy load warp quality estimation
        _mask (Optional[Mask]): lazy load mask estimation
        _glasses (Optional[Glasses]): lazy load glasses estimation
        _credibility (Optional[Credibility]): lazy load credibility estimation
        _headPose (Optional[HeadPose]): lazy load head pose estimation
        _ags (Optional[float]): lazy load ags estimation
        _transformedLandmarks5 (Optional[Landmarks68]): lazy load transformed landmarks68
        _liveness (Optional[LivenessV1]): lazy load liveness estimation
    """

    __slots__ = (
        "_estimationSettings",
        "_warp",
        "_emotions",
        "_eyes",
        "_mouthState",
        "_basicAttributes",
        "_gaze",
        "_warpQuality",
        "_headPose",
        "estimatorCollection",
        "_transformedLandmarks5",
        "_ags",
        "_descriptor",
        "_mask",
        "_glasses",
        "_liveness",
        "_credibility",
    )

    def __init__(
        self,
        coreDetection: Face,
        image: VLImage,
        estimatorCollection: FaceEstimatorsCollection,
        estimationSettings: Optional[VLFaceDetectionSettings] = None,
    ):
        """
        Init.

        Args:
            coreDetection: core detection
        """
        super().__init__(coreDetection, image)
        self._estimationSettings: VLFaceDetectionSettings = estimationSettings or VLFaceDetectionSettings()
        self._emotions: Optional[Emotions] = None
        self._eyes: Optional[EyesEstimation] = None
        self._warp: Optional[FaceWarp] = None
        self._mouthState: Optional[MouthStates] = None
        self._basicAttributes: Optional[BasicAttributes] = None
        self._gaze: Optional[GazeDirection] = None
        self._warpQuality: Optional[Quality] = None
        self._headPose: Optional[HeadPose] = None
        self._transformedLandmarks5: Optional[Landmarks5] = None
        self._ags: Optional[float] = None
        self._descriptor: Optional[FaceDescriptor] = None
        self._mask: Optional[Mask] = None
        self._glasses: Optional[Glasses] = None
        self._credibility: Optional[Credibility] = None
        self._liveness: Optional[LivenessV1] = None
        self.estimatorCollection: FaceEstimatorsCollection = estimatorCollection

    @property
    def estimationSettings(self) -> VLFaceDetectionSettings:
        """
        Get current detection settings

        Returns:
            (VLFaceDetectionSettings) Current detection settings
        """
        return self._estimationSettings

    @property
    def warp(self) -> FaceWarp:
        """
        Get warp from detection.

        Returns:
            warp
        """
        if self._warp is None:
            self._warp = self.estimatorCollection.warper.warp(self)
        return self._warp

    @property
    def headPose(self) -> HeadPose:
        """
        Get a head pose of the detection. Estimation bases on an original and a bounding box

        Returns:
            head pose
        """
        if self._headPose is None:
            imageWithFaceDetection = ImageWithFaceDetection(self.image, self.boundingBox)
            self._headPose = self.estimatorCollection.headPoseEstimator.estimateByBoundingBox(imageWithFaceDetection)
        return self._headPose

    @property
    def mouthState(self) -> MouthStates:
        """
        Get a mouth state of the detection

        Returns:
            mouth state
        """
        if self._mouthState is None:
            self._mouthState = self.estimatorCollection.mouthStateEstimator.estimate(self.warp)
        return self._mouthState

    @property
    def emotions(self) -> Emotions:
        """
        Get emotions of the detection.

        Returns:
            emotions
        """
        if self._emotions is None:
            self._emotions = self.estimatorCollection.emotionsEstimator.estimate(self.warp)
        return self._emotions

    @property
    def ags(self) -> float:
        """
        Get ags of the detection.

        Returns:
            emotions
        """
        if self._ags is None:
            self._ags = self.estimatorCollection.AGSEstimator.estimate(self)  # type: ignore
        return self._ags

    @property
    def basicAttributes(self) -> BasicAttributes:
        """
        Get all basic attributes of the detection.

        Returns:
            basic attributes (age, gender, ethnicity)
        """
        if self._basicAttributes is None:
            self._basicAttributes = self.estimatorCollection.basicAttributesEstimator.estimate(
                self.warp, estimateAge=True, estimateEthnicity=True, estimateGender=True
            )
        return self._basicAttributes

    @property
    def warpQuality(self) -> Quality:
        """
        Get quality of warped image which corresponding the detection
        Returns:
            quality
        """
        if self._warpQuality is None:
            self._warpQuality = self.estimatorCollection.warpQualityEstimator.estimate(self.warp)
        return self._warpQuality

    @property
    def mask(self) -> Mask:
        """
        Get mask existence estimation of warped image which corresponding the detection
        Returns:
            mask
        """
        if self._mask is None:
            if self._estimationSettings.estimateMaskFromDetection:
                self._mask = self.estimatorCollection.maskEstimator.estimate(self)
            else:
                self._mask = self.estimatorCollection.maskEstimator.estimate(self.warp)
        return self._mask

    @property
    def glasses(self) -> Glasses:
        """
        Get glasses existence estimation of warped image which corresponding the detection
        Returns:
            glasses
        """
        if self._glasses is None:
            self._glasses = self.estimatorCollection.glassesEstimator.estimate(self.warp)
        return self._glasses

    @property
    def credibility(self) -> Credibility:
        """
        Get credibility existence estimation of warped image which corresponding the detection
        Returns:
            credibility
        """
        if self._credibility is None:
            self._credibility = self.estimatorCollection.credibilityEstimator.estimate(self.warp)
        return self._credibility

    @property
    def descriptor(self) -> FaceDescriptor:
        """
        Get a face descriptor from warp

        Returns:
            mouth state
        """
        if self._descriptor is None:
            self._descriptor = VLWarpedImage.estimatorsCollection.descriptorEstimator.estimate(
                self.warp
            )  # type: ignore
        return self._descriptor  # type: ignore

    def _getTransformedLandmarks5(self) -> Landmarks5:
        """
        Get transformed landmarks5 for warping.

        Returns:
            landmarks5
        """
        if self._transformedLandmarks5 is None:
            warper = self.estimatorCollection.warper
            self._transformedLandmarks5 = warper.makeWarpTransformationWithLandmarks(self, "L5")  # type: ignore
        return self._transformedLandmarks5  # type: ignore

    @property
    def eyes(self) -> EyesEstimation:
        """
        Get eyes estimation of the detection.

        Returns:
            eyes estimation
        """
        if self._eyes is None:
            warpWithLandmarks = WarpWithLandmarks(self.warp, self._getTransformedLandmarks5())
            self._eyes = self.estimatorCollection.eyeEstimator.estimate(warpWithLandmarks)
        return self._eyes

    @property
    def gaze(self) -> GazeDirection:
        """
        Get gaze direction.

        Returns:
            gaze direction
        """
        if self._gaze is None:
            warpWithLandmarks5 = WarpWithLandmarks5(self.warp, self._getTransformedLandmarks5())
            self._gaze = self.estimatorCollection.gazeDirectionEstimator.estimate(warpWithLandmarks5)
        return self._gaze

    @property
    def liveness(self) -> LivenessV1:
        """
        Get livenessv1 estimation.

        Returns:
            livenessv1
        """
        if self._liveness is None:
            self._liveness = self.estimatorCollection.livenessV1Estimator.estimate(self)
        return self._liveness

    def asDict(self) -> Dict[str, Union[str, dict, list, float, tuple]]:
        """
        Convert to dict.

        Returns:
            All estimated attributes will be added to dict
        """
        res: Dict[str, Union[str, dict, list, float, tuple]] = {
            "rect": {
                "x": int(self.boundingBox.rect.x),
                "y": int(self.boundingBox.rect.y),
                "width": int(self.boundingBox.rect.width),
                "height": int(self.boundingBox.rect.height),
            }
        }
        if self._warpQuality is not None:
            res["quality"] = self.warpQuality.asDict()
        if self.landmarks5 is not None:
            res["landmarks5"] = self.landmarks5.asDict()
        if self.landmarks68 is not None:
            res["landmarks68"] = self.landmarks68.asDict()

        attributes = {}

        if self._emotions is not None:
            attributes["emotions"] = self._emotions.asDict()

        if self._eyes is not None:
            attributes["eyes_attributes"] = self._eyes.asDict()

        if self._mouthState is not None:
            attributes["mouth_attributes"] = self._mouthState.asDict()

        if self._headPose is not None:
            attributes["head_pose"] = self._headPose.asDict()

        if self._gaze is not None:
            attributes["gaze"] = self._gaze.asDict()

        if self._basicAttributes is not None:
            attributes["basic_attributes"] = self._basicAttributes.asDict()

        if self._mask is not None:
            attributes["mask"] = self._mask.asDict()

        if self._glasses is not None:
            attributes["glasses"] = self._glasses.asDict()

        if self._liveness is not None:
            attributes["liveness"] = self._liveness.asDict()
        if self._credibility is not None:
            attributes["credibility"] = self._credibility.asDict()

        res["attributes"] = attributes
        return res


class VLFaceDetector:
    """
    High level face detector. Return *VLFaceDetection* instead simple *FaceDetection*.

    Attributes:
          estimatorsCollection (FaceEstimatorsCollection): face estimator collections for new detections.
          _faceDetector (FaceDetector): face detector
          faceEngine (VLFaceEngine): face engine for detector and estimators, default *FACE_ENGINE*.
    """

    #: a global instance of FaceEngine for usual creating detectors
    faceEngine: VLFaceEngine
    #: estimators collection of class for usual creating detectors
    estimatorsCollection: FaceEstimatorsCollection

    def __init__(
        self,
        detectorType: DetectorType = DetectorType.FACE_DET_DEFAULT,
        faceEngine: Optional[VLFaceEngine] = None,
        estimationSettings: Optional[VLFaceDetectionSettings] = None,
    ):
        """
        Init.

        Args:
            detectorType: detector type
            faceEngine: face engine for detector and estimators
            estimationSettings: settings for detection
        """
        if faceEngine is None:
            if not hasattr(self, "faceEngine"):
                raise RuntimeError(f"Initialize the '{self.__class__.__name__}' first or pass faceEngine")
        else:
            self.faceEngine = faceEngine
            self.estimatorsCollection = FaceEstimatorsCollection(faceEngine=self.faceEngine)
        self._faceDetector: FaceDetector = self.faceEngine.createFaceDetector(detectorType)
        self._estimationSettings: Optional[VLFaceDetectionSettings] = estimationSettings

    @classmethod
    def initialize(cls, faceEngine: Optional[VLFaceEngine] = None) -> None:
        """
        Initialize class attributes.

        Args:
            faceEngine: face engine for detector and estimators
        """
        cls.faceEngine = faceEngine or VLFaceEngine()
        cls.estimatorsCollection = FaceEstimatorsCollection(faceEngine=cls.faceEngine)

    def detectOne(self, image: VLImage, detectArea: Optional[Rect] = None) -> Union[None, VLFaceDetection]:
        """
        Detect just one best detection on the image.

        Args:
            image: image. Format must be R8G8B8
            detectArea: rectangle area which contains face to detect. If not set will be set image.rect
        Returns:
            face detection if face is found otherwise None
        """
        detectRes = self._faceDetector.detectOne(image, detectArea, True, True)
        if detectRes is None:
            return None
        return VLFaceDetection(
            detectRes.coreEstimation, detectRes.image, self.estimatorsCollection, self._estimationSettings
        )

    def postProcessingDetectionBatch(self, detectRes: List[List[FaceDetection]]):
        """
        Post processing detection results (wrap to  VLFaceDetection)
        Args:
            detectRes: detection results

        Returns:
            VLFaceDetection's
        """
        res = []
        for imageDetections in detectRes:
            res.append(
                [
                    VLFaceDetection(
                        detection.coreEstimation,
                        detection.image,
                        self.estimatorsCollection,
                        self._estimationSettings,
                    )
                    if detection
                    else None
                    for detection in imageDetections
                ]
            )
        return res

    def postProcessing(self, detection: Optional[FaceDetection]) -> Optional[VLFaceDetection]:
        """
        Post processing detection (wrap to  VLFaceDetection)
        Args:
            detection: detection results

        Returns:
            VLFaceDetection
        """
        if detection:
            return VLFaceDetection(
                detection.coreEstimation, detection.image, self.estimatorsCollection, self._estimationSettings
            )
        return None

    def detect(
        self, images: List[Union[VLImage, ImageForDetection]], limit: int = 5
    ) -> Union[List[List[VLFaceDetection]], List[List[VLFaceDetection]]]:
        """
        Batch detect faces on images.

        Args:
            images: input images list. Format must be R8G8B8
            limit: max number of detections per input image
        Returns:
            return list of lists detection, order of detection lists is corresponding to order of input images
        """
        detectRes = self._faceDetector.detect(images, limit, True, True)
        return self.postProcessingDetectionBatch(detectRes)

    def redetectOne(self, image: Union[VLImage, VLFaceDetection], bBox: Rect) -> Union[VLFaceDetection, None]:
        """
        Redetect faces on an image. If VLFaceDetection is provided, only VLImage from that object will be used.

        Args:
            image: input image. Image format must be R8G8B8
            bBox: bounding box
        Returns:
            return detection or None if face not found
        """
        if isinstance(image, VLFaceDetection):
            imageForRedetct = image.image
        else:
            imageForRedetct = image
        redetection: Union[None, FaceDetection] = self._faceDetector.redetectOne(
            imageForRedetct, bBox=bBox, detect5Landmarks=True, detect68Landmarks=True
        )
        return self.postProcessing(redetection)

    def redetect(
        self,
        imagesAndBBoxes: List[ImageForRedetection],
    ) -> Union[List[List[Union[VLFaceDetection, None]]], List[List[Union[VLFaceDetection, None]]]]:
        """
        Redetect faces on images.

        Args:
            imagesAndBBoxes: input tuples: [(VLImage, [bBox1, bBox2])]. Image format must be R8G8B8

        Returns:
            detections: [[redetection]]. Order of detection lists is corresponding to order of input images.
                Order of detections is corresponding to order of input bounding boxes.
        """

        redetections = self._faceDetector.redetect(imagesAndBBoxes, True, True)
        return self.postProcessingDetectionBatch(redetections)


class VLWarpedImage(FaceWarpedImage):
    """
    High level sample object.

    Attributes:

        _emotions (Optional[Emotions]): lazy load emotions estimations
        _mouthState (Optional[MouthStates]): lazy load mouth state estimation
        _basicAttributes (Optional[BasicAttributes]): lazy load basic attribute estimation
        _warpQuality (Optional[Quality]): lazy load warp quality estimation
        _mask (Optional[Mask]): lazy load mask estimation
        _glasses (Optional[Glasses]): lazy load glasses estimation
        _credibility (Optional[Credibility]): lazy load credibility estimation
    """

    #: estimators collection of class for usual creating detectors
    estimatorsCollection: FaceEstimatorsCollection

    __slots__ = (
        "_emotions",
        "_mouthState",
        "_basicAttributes",
        "_warpQuality",
        "_descriptor",
        "_mask",
        "_glasses",
        "_credibility",
    )

    def __init__(
        self,
        body: Union[bytes, ndarray, PilImage, CoreImage, VLImage],
        filename: str = "",
        colorFormat: Optional[ColorFormat] = None,
    ):
        super().__init__(body=body, filename=filename, colorFormat=colorFormat)
        self._emotions: Optional[Emotions] = None
        self._eyes: Optional[EyesEstimation] = None
        self._mouthState: Optional[MouthStates] = None
        self._basicAttributes: Optional[BasicAttributes] = None
        self._warpQuality: Optional[Quality] = None
        self._descriptor: Optional[FaceDescriptor] = None
        self._mask: Optional[Mask] = None
        self._glasses: Optional[Glasses] = None
        self._credibility: Optional[Credibility] = None

    @classmethod
    def initialize(cls, estimatorsCollection: Optional[FaceEstimatorsCollection] = None) -> None:
        """
        Initialize class attributes.

        Args:
            estimatorsCollection: face estimators collection
        """
        cls.estimatorsCollection = estimatorsCollection or FaceEstimatorsCollection()

    @property
    def mouthState(self) -> MouthStates:
        """
        Get a mouth state of the detection

        Returns:
            mouth state
        """
        if self._mouthState is None:
            self._mouthState = VLWarpedImage.estimatorsCollection.mouthStateEstimator.estimate(self)
        return self._mouthState

    @property
    def descriptor(self) -> FaceDescriptor:
        """
        Get a face descriptor from warp

        Returns:
            mouth state
        """
        if self._descriptor is None:
            self._descriptor = VLWarpedImage.estimatorsCollection.descriptorEstimator.estimate(self)  # type: ignore
        return self._descriptor  # type: ignore

    @property
    def emotions(self) -> Emotions:
        """
        Get emotions of the detection.

        Returns:
            emotions
        """
        if self._emotions is None:
            self._emotions = VLWarpedImage.estimatorsCollection.emotionsEstimator.estimate(self)
        return self._emotions

    @property
    def basicAttributes(self) -> BasicAttributes:
        """
        Get all basic attributes of the detection.

        Returns:
            basic attributes (age, gender, ethnicity)
        """
        if self._basicAttributes is None:
            estimator = VLWarpedImage.estimatorsCollection.basicAttributesEstimator
            self._basicAttributes = estimator.estimate(
                self, estimateAge=True, estimateEthnicity=True, estimateGender=True
            )
        return self._basicAttributes

    @property
    def warpQuality(self) -> Quality:
        """
        Get quality of warped image which corresponding the detection
        Returns:
            quality
        """
        if self._warpQuality is None:
            self._warpQuality = VLWarpedImage.estimatorsCollection.warpQualityEstimator.estimate(self)
        return self._warpQuality

    @property
    def mask(self) -> Mask:
        """
        Get mask of warped image which corresponding the detection
        Returns:
            mask
        """
        if self._mask is None:
            self._mask = VLWarpedImage.estimatorsCollection.maskEstimator.estimate(self)
        return self._mask

    @property
    def glasses(self) -> Glasses:
        """
        Get glasses of warped image which corresponding the detection
        Returns:
            glasses
        """
        if self._glasses is None:
            self._glasses = VLWarpedImage.estimatorsCollection.glassesEstimator.estimate(self)
        return self._glasses

    @property
    def credibility(self) -> Credibility:
        """
        Get ccredibility of warped image which corresponding the detection
        Returns:
            credibility
        """
        if self._credibility is None:
            self._credibility = VLWarpedImage.estimatorsCollection.credibilityEstimator.estimate(self)
        return self._credibility

    def asDict(self) -> Dict[str, Dict[str, float]]:
        """
        Convert to dict.

        Returns:
            All estimated attributes will be added to dict
        """
        res = {}
        if self._warpQuality is not None:
            res["quality"] = self.warpQuality.asDict()

        attributes = {}

        if self._emotions is not None:
            attributes["emotions"] = self._emotions.asDict()

        if self._eyes is not None:
            attributes["eyes_attributes"] = self._eyes.asDict()

        if self._mouthState is not None:
            attributes["mouth_attributes"] = self._mouthState.asDict()

        if self._basicAttributes is not None:
            attributes["basic_attributes"] = self._basicAttributes.asDict()

        if self._mask is not None:
            attributes["mask"] = self._mask.asDict()

        if self._glasses is not None:
            attributes["glasses"] = self._glasses.asDict()

        if self._credibility is not None:
            attributes["credibility"] = self._credibility.asDict()

        res["attributes"] = attributes
        return res

    @property
    def warp(self) -> FaceWarpedImage:
        """
        Support VLFaceDetection interface.

        Returns:
            self
        """
        return self
