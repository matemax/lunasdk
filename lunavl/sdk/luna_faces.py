"""
Module realize hight level api for estimate face attributes
"""
from typing import Optional, Union, List, Dict

from FaceEngine import Face  # pylint: disable=E0611,E0401
from FaceEngine import Image as CoreImage  # pylint: disable=E0611,E0401
from PIL.Image import Image as PilImage
from numpy import ndarray
from .estimator_collections import FaceEstimatorsCollection
from .estimators.face_estimators.basic_attributes import BasicAttributes
from .estimators.face_estimators.emotions import Emotions
from .estimators.face_estimators.eyes import EyesEstimation, GazeDirection
from .estimators.face_estimators.face_descriptor import FaceDescriptor
from .estimators.face_estimators.head_pose import HeadPose
from .estimators.face_estimators.mouth_state import MouthStates
from .estimators.face_estimators.warp_quality import Quality
from .estimators.face_estimators.mask import Mask
from .estimators.face_estimators.glasses import Glasses
from .estimators.face_estimators.credibility_check import CredibilityCheck
from .estimators.face_estimators.facewarper import FaceWarp, FaceWarpedImage
from .faceengine.engine import VLFaceEngine
from .detectors.facedetector import FaceDetection, FaceDetector, Landmarks5
from .detectors.base import ImageForDetection, ImageForRedetection
from .faceengine.setting_provider import DetectorType
from .image_utils.geometry import Rect
from .image_utils.image import VLImage, ColorFormat


class VLFaceDetectionSettings:
    """
    Settings for detection

    Attributes:
        estimateMaskFromDetection: estimate mask from detection or warp

    """

    __slots__ = "_estimateMaskFromDetection"

    def __init__(self, estimateMaskFromDetection: bool = True):
        """
        Init settings

        Args:
            estimateMaskFromDetection: estimate from detection or warp
        """
        self._estimateMaskFromDetection = estimateMaskFromDetection

    @property
    def estimateMaskFromDetection(self) -> bool:
        """Get current settings for mask estimation"""
        return self._estimateMaskFromDetection


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
        _credibilityCheck (Optional[CredibilityCheck]): lazy load credibility check estimation
        _headPose (Optional[HeadPose]): lazy load head pose estimation
        _ags (Optional[float]): lazy load ags estimation
        _transformedLandmarks5 (Optional[Landmarks68]): lazy load transformed landmarks68

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
        "_credibilityCheck",
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
        self._credibilityCheck: Optional[CredibilityCheck] = None
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
            self._headPose = self.estimatorCollection.headPoseEstimator.estimateByBoundingBox(
                self.boundingBox, self.image
            )
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
    def credibilityCheck(self) -> CredibilityCheck:
        """
        Get credibility check existence estimation of warped image which corresponding the detection
        Returns:
            credibilityCheck
        """
        if self._credibilityCheck is None:
            self._credibilityCheck = self.estimatorCollection.credibilityCheckEstimator.estimate(self.warp)
        return self._credibilityCheck

    @property
    def descriptor(self) -> FaceDescriptor:
        """
        Get a face descriptor from warp

        Returns:
            mouth state
        """
        if self._descriptor is None:
            self._descriptor = VLWarpedImage.estimatorsCollection.descriptorEstimator.estimate(self.warp)
        return self._descriptor

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
            self._eyes = self.estimatorCollection.eyeEstimator.estimate(self._getTransformedLandmarks5(), self.warp)
        return self._eyes

    @property
    def gaze(self) -> GazeDirection:
        """
        Get gaze direction.

        Returns:
            gaze direction
        """
        if self._gaze is None:
            self._gaze = self.estimatorCollection.gazeDirectionEstimator.estimate(
                self._getTransformedLandmarks5(), self.warp
            )
        return self._gaze

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

        if self._credibilityCheck is not None:
            attributes["credibility_check"] = self._credibilityCheck.asDict()

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
    faceEngine: VLFaceEngine = VLFaceEngine()
    #: estimators collection of class for usual creating detectors
    estimatorsCollection: FaceEstimatorsCollection = FaceEstimatorsCollection(faceEngine=faceEngine)

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
        """
        if faceEngine is not None:
            self.faceEngine = faceEngine
            self.estimatorsCollection = FaceEstimatorsCollection(faceEngine=self.faceEngine)
        self._faceDetector: FaceDetector = self.faceEngine.createFaceDetector(detectorType)
        self._estimationSettings: Optional[VLFaceDetectionSettings] = estimationSettings

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

    def detect(self, images: List[Union[VLImage, ImageForDetection]], limit: int = 5) -> List[List[VLFaceDetection]]:
        """
        Batch detect faces on images.

        Args:
            images: input images list. Format must be R8G8B8
            limit: max number of detections per input image
        Returns:
            return list of lists detection, order of detection lists is corresponding to order of input images
        """
        detectRes = self._faceDetector.detect(images, limit, True, True)
        res = []
        for imageNumber, image in enumerate(images):
            res.append(
                [
                    VLFaceDetection(
                        detectRes.coreEstimation,
                        image if isinstance(image, VLImage) else image.image,
                        self.estimatorsCollection,
                        self._estimationSettings,
                    )
                    for detectRes in detectRes[imageNumber]
                ]
            )
        return res

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
        if redetection:
            return VLFaceDetection(
                redetection.coreEstimation, redetection.image, self.estimatorsCollection, self._estimationSettings
            )
        return None

    def redetect(self, imagesAndBBoxes: List[ImageForRedetection]) -> List[List[Union[VLFaceDetection, None]]]:
        """
        Redetect faces on images.

        Args:
            imagesAndBBoxes: input tuples: [(VLImage, [bBox1, bBox2])]. Image format must be R8G8B8

        Returns:
            detections: [[redetection]]. Order of detection lists is corresponding to order of input images.
                Order of detections is corresponding to order of input bounding boxes.
        """

        redetections: List[List[Union[FaceDetection, None]]] = self._faceDetector.redetect(imagesAndBBoxes, True, True)
        res = []
        for redetectionsOfImage in redetections:
            imageRes = [
                VLFaceDetection(
                    redetection.coreEstimation, redetection.image, self.estimatorsCollection, self._estimationSettings
                )
                if redetection
                else None
                for redetection in redetectionsOfImage
            ]
            res.append(imageRes)
        return res


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
        _credibilityCheck (Optional[CredibilityCheck]): lazy load credibility check estimation
    """

    __slots__ = (
        "_emotions",
        "_mouthState",
        "_basicAttributes",
        "_warpQuality",
        "_descriptor",
        "_mask",
        "_glasses",
        "_credibilityCheck",
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
        self._credibilityCheck: Optional[CredibilityCheck] = None

    #: estimators collection of class for usual creating detectors
    estimatorsCollection: FaceEstimatorsCollection = FaceEstimatorsCollection(faceEngine=VLFaceEngine())

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
            self._descriptor = VLWarpedImage.estimatorsCollection.descriptorEstimator.estimate(self)
        return self._descriptor

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
    def credibilityCheck(self) -> CredibilityCheck:
        """
        Get credibility check of warped image which corresponding the detection
        Returns:
            credibilityCheck
        """
        if self._credibilityCheck is None:
            self._credibilityCheck = VLWarpedImage.estimatorsCollection.credibilityCheckEstimator.estimate(self)
        return self._credibilityCheck

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

        if self._credibilityCheck is not None:
            attributes["credibility_check"] = self._credibilityCheck.asDict()

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
