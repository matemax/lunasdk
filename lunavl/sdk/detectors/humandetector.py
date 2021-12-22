"""
Module contains function for detection human bodies on images.
"""
from functools import partial
from typing import Optional, Union, List, Dict, Any

from FaceEngine import (
    Human,
    HumanLandmarks17 as CoreLandmarks17,
    Detection,
    HumanDetectionType,
)  # pylint: disable=E0611,E0401

from .base import (
    ImageForDetection,
    ImageForRedetection,
    BaseDetection,
    assertImageForDetection,
    getArgsForCoreDetectorForImages,
    validateBatchDetectInput,
    getArgsForCoreRedetect,
    validateReDetectInput,
)
from ..async_task import AsyncTask
from ..base import LandmarksWithScore
from ..errors.errors import LunaVLError
from ..errors.exceptions import CoreExceptionWrap, assertError
from ..image_utils.geometry import Rect
from ..image_utils.image import VLImage


def _createCoreHumans(image: ImageForRedetection) -> List[Human]:
    """
    Create core humans for redetection
    Args:
        image: image and bounding boxes for redetection

    Returns:
        Human object list. one object for one bbox
    """
    humans = [Human() for _ in range(len(image.bBoxes))]
    for index, human in enumerate(humans):
        human.img = image.image.coreImage
        human.detection.setRawRect(image.bBoxes[index].coreRectF)
        human.detection.setScore(1)
    return humans


class Landmarks17(LandmarksWithScore):
    """
    Landmarks17
    """

    #  pylint: disable=W0235
    def __init__(self, coreLandmark17: CoreLandmarks17):
        """
        Init

        Args:
            coreLandmark17: core landmarks
        """
        super().__init__(coreLandmark17)


class HumanDetection(BaseDetection):
    """
    Attributes:
        landmarks17 (Optional[Landmarks17]): optional landmarks17
    """

    __slots__ = ("landmarks17",)

    def __init__(self, coreDetection: Human, image: VLImage):
        """
        Init.

        Args:
            coreDetection: core detection
        """
        super().__init__(coreDetection, image)

        if coreDetection.landmarks17_opt.isValid():
            self.landmarks17: Optional[Landmarks17] = Landmarks17(coreDetection.landmarks17_opt.value())
        else:
            self.landmarks17 = None

    def asDict(self) -> Dict[str, Any]:
        """
        Convert human detection to dict (json).

        Returns:
            dict. required keys: 'rect', 'score'. optional keys: 'landmarks5', 'landmarks68'
        """
        res = super().asDict()
        if self.landmarks17 is not None:
            res["landmarks17"] = self.landmarks17.asDict()
        return res


def createHumanDetection(image: VLImage, detection: Detection, landmarks17: Optional[CoreLandmarks17]):
    human = Human()
    human.img = image.coreImage
    human.detection = detection
    if landmarks17:
        human.landmarks17_opt.set(landmarks17)
    return HumanDetection(human, image)


def collectDetectionsResult(
    fsdkDetectRes,
    images: Union[List[Union[VLImage, ImageForDetection]], List[ImageForRedetection]],
    detectLandmarks: bool = True,
):
    """
    Collect detection results from core reply and prepare human detections
    Args:
        fsdkDetectRes: fsdk (re)detect results
        images: incoming images
        detectLandmarks: detect body landmarks or not
    Returns:
        return list of lists detection, order of detection lists is corresponding to order input images
    """
    res = []
    for imageIdx in range(fsdkDetectRes.getSize()):
        imagesDetections = []
        detections = fsdkDetectRes.getDetections(imageIdx)
        landmarks17Array = fsdkDetectRes.getLandmarks17(imageIdx)
        image = images[imageIdx]
        vlImage = image if isinstance(image, VLImage) else image.image
        for detectionIdx, detection in enumerate(detections):
            landMarks = landmarks17Array[detectionIdx] if detectLandmarks else None
            if detection.isValid():
                humanDetection = createHumanDetection(vlImage, detection, landMarks)
            else:
                humanDetection = None
            imagesDetections.append(humanDetection)
        res.append(imagesDetections)
    return res


def postProcessing(error, detectRes, image):
    assertError(error)

    detections = detectRes.getDetections(0)
    landmarks17Array = detectRes.getLandmarks17(0)

    isReplyNotAssumesDetection = detectRes.getSize() == 1
    if isReplyNotAssumesDetection:
        isDetectionExistsNValid = len(detections) != 0 and detections[0].isValid()
        if not isDetectionExistsNValid:
            return None

    landmarks17 = landmarks17Array[0] if landmarks17Array else None
    humanDetection = createHumanDetection(image, detections[0], landmarks17)
    return humanDetection


def postProcessingRedetect(error, detectRes, image):
    assertError(error)
    if detectRes.isValid():
        return HumanDetection(detectRes, image)
    return None


def postProcessingBatch(error, fsdkDetectRes, images, detectLandmarks):
    assertError(error)
    return collectDetectionsResult(fsdkDetectRes, images=images, detectLandmarks=detectLandmarks)


DetectOneResult = Union[None, HumanDetection]
DetectResult = List[List[HumanDetection]]
RedetectBatchResult = List[List[Union[HumanDetection, None]]]
RedetectResult = Union[None, HumanDetection]


class HumanDetector:
    """
    Human body detector.

    Attributes:
        _detector (IDetectorPtr): core detector
    """

    __slots__ = ("_detector",)

    def __init__(self, detectorPtr):
        self._detector = detectorPtr

    @staticmethod
    def _getDetectionType(detectLandmarks: bool = True) -> HumanDetectionType:
        """
        Get  core detection type

        Args:
            detectLandmarks: detect or not landmarks
        Returns:
            detection type
        """
        toDetect = HumanDetectionType.HDT_BOX

        if detectLandmarks:
            toDetect = HumanDetectionType.HDT_ALL
        return toDetect

    @CoreExceptionWrap(LunaVLError.DetectHumanError)
    def detectOne(
        self,
        image: VLImage,
        detectArea: Optional[Rect] = None,
        limit: int = 5,
        detectLandmarks: bool = True,
        asyncEstimate=False,
    ) -> Union[DetectOneResult, AsyncTask[DetectOneResult]]:
        """
        Detect just one best detection on the image.

        Args:
            image: image. Format must be R8G8B8
            detectArea: rectangle area which contains human to detect. If not set will be set image.rect
            limit: max number of detections for input image
            detectLandmarks: detect or not landmarks
            asyncEstimate: estimate or run estimation in background
        Returns:
            human detection if human is found otherwise None
        Raises:
            LunaSDKException: if detectOne is failed or image format has wrong the format
        """
        assertImageForDetection(image)
        detectionType = self._getDetectionType(detectLandmarks)

        if detectArea is None:
            forDetection = ImageForDetection(image=image, detectArea=image.rect)
        else:
            forDetection = ImageForDetection(image=image, detectArea=detectArea)
        imgs, detectAreas = getArgsForCoreDetectorForImages([forDetection])
        if asyncEstimate:
            task = self._detector.asyncDetect(imgs, detectAreas, limit, detectionType)
            return AsyncTask(task, postProcessing=partial(postProcessing, image=image))
        error, detectRes = self._detector.detect(imgs, detectAreas, limit, detectionType)
        return postProcessing(error, detectRes, image)

    @CoreExceptionWrap(LunaVLError.DetectHumansError)
    def detect(
        self,
        images: List[Union[VLImage, ImageForDetection]],
        limit: int = 5,
        detectLandmarks: bool = True,
        asyncEstimate=False,
    ) -> Union[DetectResult, AsyncTask[DetectResult]]:
        """
        Batch detect human bodies on images.

        Args:
            images: input images list. Format must be R8G8B8
            limit: max number of detections per input image
            detectLandmarks: detect or not landmarks
            asyncEstimate: estimate or run estimation in background
        Returns:
            asyncEstimate is False: return list of lists detection, order of detection lists is corresponding
                                    to order input images
            asyncEstimate isTrue:  async task
        """
        coreImages, detectAreas = getArgsForCoreDetectorForImages(images)
        detectionType = self._getDetectionType(detectLandmarks)
        validateBatchDetectInput(self._detector, coreImages, detectAreas)
        if asyncEstimate:
            task = self._detector.asyncDetect(coreImages, detectAreas, limit, detectionType)
            return AsyncTask(task, partial(postProcessingBatch, images=images, detectLandmarks=detectLandmarks))
        error, fsdkDetectRes = self._detector.detect(coreImages, detectAreas, limit, detectionType)
        return postProcessingBatch(error, fsdkDetectRes, images, detectLandmarks)

    @CoreExceptionWrap(LunaVLError.DetectHumansError)
    def redetectOne(  # noqa: F811
        self,
        image: VLImage,
        bBox: Union[Rect, HumanDetection],
        detectLandmarks: bool = True,
        asyncEstimate=False,
    ) -> Union[RedetectResult, AsyncTask[RedetectResult]]:
        """
        Redetect human body on an image in area, restricted with image.bBox, bBox or detection.

        Args:
            image: image with a bounding box, or just VLImage. If VLImage provided, one of bBox or detection
                should be defined.
            bBox: detection bounding box
            detectLandmarks: detect or not landmarks

        Returns:
            detection if human body found otherwise None if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException if an error occurs
        """
        assertImageForDetection(image)
        if isinstance(bBox, Rect):
            coreBBox = Detection(bBox.coreRectF, 1.0)
        else:
            coreBBox = bBox.coreEstimation.detection
        if asyncEstimate:
            task = self._detector.asyncRedetectOne(image.coreImage, coreBBox, self._getDetectionType(detectLandmarks))
            return AsyncTask(task, partial(postProcessingRedetect, image=image))
        error, detectRes = self._detector.redetectOne(
            image.coreImage, coreBBox, self._getDetectionType(detectLandmarks)
        )
        return postProcessingRedetect(error, detectRes, image)

    @CoreExceptionWrap(LunaVLError.DetectHumansError)
    def redetect(
        self,
        images: List[ImageForRedetection],
        detectLandmarks: bool = True,
        asyncEstimate=False,
    ) -> Union[RedetectBatchResult, AsyncTask[RedetectBatchResult]]:
        """
        Redetect human on each image.image in area, restricted with image.bBox.

        Args:
            images: images with a bounding boxes,
            asyncEstimate: estimate or run estimation in background
            detectLandmarks: detect or not landmarks

        Returns:
            detections if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException if an error occurs, context contains all errors
        """
        coreImages, detectAreas = getArgsForCoreRedetect(images)
        validateReDetectInput(self._detector, coreImages, detectAreas)
        if asyncEstimate:
            task = self._detector.asyncRedetect(coreImages, detectAreas, self._getDetectionType(detectLandmarks))
            pProcessing = partial(postProcessingBatch, images=images, detectLandmarks=detectLandmarks)
            return AsyncTask(task, pProcessing)
        error, fsdkDetectRes = self._detector.redetect(coreImages, detectAreas, self._getDetectionType(detectLandmarks))
        return postProcessingBatch(error, fsdkDetectRes, images=images, detectLandmarks=detectLandmarks)
