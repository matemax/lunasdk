"""Module realize hight level api for estimate face attributes
"""
from collections import defaultdict
from typing import Optional, Union, List, Dict, Tuple, Iterable

from FaceEngine import Face, DetectionType, dtBBox, DetectionFloat  # pylint: disable=E0611,E0401
from FaceEngine import Image as CoreImage  # pylint: disable=E0611,E0401
from numpy.ma import array

from lunavl.sdk.estimator_collections import FaceEstimatorsCollection
from lunavl.sdk.estimators.face_estimators.basic_attributes import BasicAttributes
from lunavl.sdk.estimators.face_estimators.emotions import Emotions
from lunavl.sdk.estimators.face_estimators.eyes import EyesEstimation, GazeDirection
from lunavl.sdk.estimators.face_estimators.face_descriptor import FaceDescriptor
from lunavl.sdk.estimators.face_estimators.head_pose import HeadPose
from lunavl.sdk.estimators.face_estimators.mouth_state import MouthStates
from lunavl.sdk.estimators.face_estimators.warp_quality import Quality
from lunavl.sdk.estimators.face_estimators.warper import Warp, WarpedImage
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.facedetector import FaceDetection, ImageForDetection, FaceDetector, Landmarks5, \
    ImageForRedetection
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage


class VLFaceDetection(FaceDetection):
    """
    High level detection object.
    Attributes:

        estimatorCollection (FaceEstimatorsCollection): collection of estimators
        _emotions (Optional[Emotions]): lazy load emotions estimations
        _eyes (Optional[EyesEstimation]): lazy load eye estimations
        _mouthState (Optional[MouthStates]): lazy load mouth state estimation
        _basicAttributes (Optional[BasicAttributes]): lazy load basic attribute estimation
        _gaze (Optional[GazeEstimation]): lazy load gaze direction estimation
        _warpQuality (Optional[Quality]): lazy load warp quality estimation
        _headPose (Optional[HeadPose]): lazy load head pose estimation
        _ags (Optional[float]): lazy load ags estimation
        _transformedLandmarks5 (Optional[Landmarks68]): lazy load transformed landmarks68

    """

    __slots__ = (
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
    )

    def __init__(self, coreDetection: Face, image: VLImage, estimatorCollection: FaceEstimatorsCollection):
        """
        Init.

        Args:
            coreDetection: core detection
        """
        super().__init__(coreDetection, image)
        self._emotions: Optional[Emotions] = None
        self._eyes: Optional[EyesEstimation] = None
        self._warp: Optional[Warp] = None
        self._mouthState: Optional[MouthStates] = None
        self._basicAttributes: Optional[BasicAttributes] = None
        self._gaze: Optional[GazeDirection] = None
        self._warpQuality: Optional[Quality] = None
        self._headPose: Optional[HeadPose] = None
        self._transformedLandmarks5: Optional[Landmarks5] = None
        self._ags = None
        self._descriptor = None
        self.estimatorCollection = estimatorCollection

    @property
    def warp(self) -> Warp:
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
            self._headPose = self.estimatorCollection.headPoseEstimator.estimateByBoundingBox(self.boundingBox,
                                                                                              self.image)
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
            self._ags = self.estimatorCollection.AGSEstimator.estimate(self)
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
            self._transformedLandmarks5 = self.estimatorCollection.warper.makeWarpTransformationWithLandmarks(
                self, "L5"
            )
        return self._transformedLandmarks5

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
            self._gaze = self.estimatorCollection.gazeDirectionEstimator.estimate(self._getTransformedLandmarks5(),
                                                                                  self.warp)
        return self._gaze

    def asDict(self) -> Dict[str, Union[dict, list, float]]:
        """
        Convert to dict.

        Returns:
            All estimated attributes will be added to dict
        """
        res = {
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
            self, detectorType: DetectorType = DetectorType.FACE_DET_DEFAULT, faceEngine: Optional[VLFaceEngine] = None
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

    def detectOne(self, image: VLImage, detectArea: Optional[Rect] = None) -> Union[None, VLFaceDetection]:
        """
        Detect just one best detection on the image.

        Args:
            image: image. Format must be R8G8B8 (todo check)
            detectArea: rectangle area which contains face to detect. If not set will be set image.rect
        Returns:
            face detection if face is found otherwise None
        """
        detectRes = self._faceDetector.detectOne(image, detectArea, True, True)
        if detectRes is None:
            return None
        return VLFaceDetection(detectRes.coreEstimation, detectRes.image, self.estimatorsCollection)

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
                    VLFaceDetection(detectRes.coreEstimation, image if isinstance(image, VLImage) else image.image,
                                    self.estimatorsCollection)
                    for detectRes in detectRes[imageNumber]
                ]
            )
        return res

    def redetectOne(self, image: Union[VLImage, VLFaceDetection], bBoxes: List[Rect]) -> List[VLFaceDetection]:
        """
        Redetect faces on an image. If VLFaceDetection is provided, only VLImage from that object will be used.

        Args:
            image: input image. Image format must be R8G8B8
            bBoxes: bounding boxes
        Returns:
            return list of detection, order of detections is corresponding to order input bounding boxes
        """
        if isinstance(image, VLFaceDetection):
            image = VLFaceDetection.image
        redetections: Iterable[FaceDetection] = (self._faceDetector.redetectOne(
            ImageForRedetection(image, bBox),
            detect5Landmarks=True,
            detect68Landmarks=True
        ) for bBox in bBoxes)

        res = [VLFaceDetection(redetection.coreEstimation, redetection.image, self.estimatorsCollection)
               for redetection in redetections]
        return res

    def redetect(self, imagesAndBBoxes: List[Tuple[VLImage, List[Rect]]]) -> List[List[VLFaceDetection]]:
        """
        Redetect faces on images.

        Args:
            imagesAndBBoxes: input tuples: [(VLImage, [bBox1, bBox2])]. Image format must be R8G8B8

        Returns:
            detections: [[redetection]]. Order of detection lists is corresponding to order of input images.
                Order of detections is corresponding to order of input bounding boxes.
        """
        # [image1[bbox1, bbox2], image2[bbox3]] -> [image1[bbox1], image1[bbox2], image2[bbox3]]
        flatToImgIdx: Dict[int, int] = {}
        flatImages: List[ImageForRedetection] = []
        for imageIdx, (image, bBoxes) in enumerate(imagesAndBBoxes):
            for bBox in bBoxes:
                newImage = ImageForRedetection(image, bBox)
                flatToImgIdx[len(flatImages)] = imageIdx
                flatImages.append(newImage)

        redetections: List[FaceDetection] = self._faceDetector.redetect(flatImages, True, True)

        # [redetection1, redetection2, redetection3] -> [[redetection1, redetection2], [redetection3]]
        res = [[] for _ in range(len(imagesAndBBoxes))]
        for detIdx, redetection in enumerate(redetections):
            res[flatToImgIdx[detIdx]].append(VLFaceDetection(redetection.coreEstimation, redetection.image,
                                                             self.estimatorsCollection))
        return res


class VLWarpedImage(WarpedImage):
    """
    High level sample object.

    Attributes:

        _emotions (Optional[Emotions]): lazy load emotions estimations
        _mouthState (Optional[MouthStates]): lazy load mouth state estimation
        _basicAttributes (Optional[BasicAttributes]): lazy load basic attribute estimation
        _warpQuality (Optional[Quality]): lazy load warp quality estimation
    """

    __slots__ = ("_emotions", "_mouthState", "_basicAttributes", "_warpQuality", "_descriptor")

    def __init__(self, body: Union[bytes, array, CoreImage], filename: str = "", vlImage: Optional[VLImage] = None):
        super().__init__(body, filename, vlImage)
        self._emotions: Optional[Emotions] = None
        self._eyes: Optional[EyesEstimation] = None
        self._mouthState: Optional[MouthStates] = None
        self._basicAttributes: Optional[BasicAttributes] = None
        self._warpQuality: Optional[Quality] = None
        self._descriptor: Optional[FaceDescriptor] = None

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

    def asDict(self) -> Dict[str, Union[dict, list, float]]:
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

        res["attributes"] = attributes
        return res

    @property
    def warp(self) -> Warp:
        """
        Support VLFaceDetection interface.

        Returns:
            self
        """
        return self
