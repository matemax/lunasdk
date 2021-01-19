"""
Module contains function for detection human bodies on images.
"""
from typing import Optional, Union, List, Dict, Any

from FaceEngine import (
    HumanDetectionType,
    Human,
    HumanLandmarks17 as CoreLandmarks17,
    Detection,
    Image as CoreImage,
    Rect as CoreRectI,
)  # pylint: disable=E0611,E0401

from .base import (
    ImageForDetection,
    ImageForRedetection,
    BaseDetection,
    assertImageForDetection,
    getArgsForCoreDetectorForImages,
    collectAndRaiseErrorIfOccurred,
)
from ..base import LandmarksWithScore
from ..errors.errors import LunaVLError
from ..errors.exceptions import CoreExceptionWrap, assertError, LunaSDKException
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
        self, image: VLImage, detectArea: Optional[Rect] = None, detectLandmarks: bool = True
    ) -> Union[None, HumanDetection]:
        """
        Detect just one best detection on the image.

        Args:
            image: image. Format must be R8G8B8
            detectArea: rectangle area which contains human to detect. If not set will be set image.rect
            detectLandmarks: detect or not landmarks
        Returns:
            human detection if human is found otherwise None
        Raises:
            LunaSDKException: if detectOne is failed or image format has wrong  the format
        """
        assertImageForDetection(image)
        detectionType = self._getDetectionType(detectLandmarks)

        if detectArea is None:
            forDetection = ImageForDetection(image=image, detectArea=image.rect)
        else:
            forDetection = ImageForDetection(image=image, detectArea=detectArea)
        imgs, detectAreas = getArgsForCoreDetectorForImages([forDetection])
        error, detectRes = self._detector.detect(imgs, detectAreas, 1, detectionType)
        assertError(error)

        detections = detectRes.getDetections(0)
        landmarks17Array = detectRes.getLandmarks17(0)

        isReplyNotAssumesDetection = detectRes.getSize() == 1
        if isReplyNotAssumesDetection:
            isDetectionExists = len(detections) != 0
            isDetectionExistsNValid = isDetectionExists and detections[0].isValid()
            if not isDetectionExistsNValid:
                return None

        human = Human()
        human.img = image.coreImage
        human.detection = detections[0]
        if landmarks17Array and landmarks17Array[0] is not None:
            human.landmarks17_opt.set(landmarks17Array[0])
        return HumanDetection(human, image)

    @CoreExceptionWrap(LunaVLError.DetectHumansError)
    def detect(
        self, images: List[Union[VLImage, ImageForDetection]], limit: int = 5, detectLandmarks: bool = True
    ) -> List[List[HumanDetection]]:
        """
        Batch detect human bodies on images.

        Args:
            images: input images list. Format must be R8G8B8
            limit: max number of detections per input image
            detectLandmarks: detect or not landmarks
        Returns:
            return list of lists detection, order of detection lists is corresponding to order input images
        """

        def getSingleError(image: CoreImage, detectArea: CoreRectI):
            errorOne, _ = self._detector.detect([image], [detectArea], 1, detectionType)
            return errorOne

        coreImages, detectAreas = getArgsForCoreDetectorForImages(images)
        detectionType = self._getDetectionType(detectLandmarks)

        fsdkErrorRes, fsdkDetectRes = self._detector.detect(coreImages, detectAreas, limit, detectionType)
        collectAndRaiseErrorIfOccurred(fsdkErrorRes, coreImages, detectAreas, getSingleError)

        res = []
        for imageIdx in range(fsdkDetectRes.getSize()):
            imagesDetections = []
            detections = fsdkDetectRes.getDetections(imageIdx)
            landmarks17Array = fsdkDetectRes.getLandmarks17(imageIdx)

            for detection, landmarks17 in zip(detections, landmarks17Array):
                human = Human()
                human.img = coreImages[imageIdx]
                human.detection = detection
                if landmarks17:
                    human.landmarks17_opt.set(landmarks17)
                imagesDetections.append(human)

            image = images[imageIdx]
            vlImage = image if isinstance(image, VLImage) else image.image
            res.append([HumanDetection(human, vlImage) for human in imagesDetections])

        return res

    @CoreExceptionWrap(LunaVLError.DetectHumansError)
    def redetectOne(  # noqa: F811
        self, image: VLImage, bBox: Union[Rect, HumanDetection]
    ) -> Union[None, HumanDetection]:
        """
        Redetect human body on an image in area, restricted with image.bBox, bBox or detection.

        Args:
            image: image with a bounding box, or just VLImage. If VLImage provided, one of bBox or detection
                should be defined.
            bBox: detection bounding box

        Returns:
            detection if human body found otherwise None
        Raises:
            LunaSDKException if an error occurs
        """
        assertImageForDetection(image)
        if isinstance(bBox, Rect):
            coreBBox = Detection(bBox.coreRectF, 1.0)
        else:
            coreBBox = bBox.coreEstimation.detection

        error, detectRes = self._detector.redetectOne(image.coreImage, coreBBox)

        assertError(error)
        if detectRes.isValid():
            return HumanDetection(detectRes, image)
        return None

    @CoreExceptionWrap(LunaVLError.DetectHumansError)
    def redetect(self, images: List[ImageForRedetection]) -> List[List[Union[HumanDetection, None]]]:
        """
        Redetect human on each image.image in area, restricted with image.bBox.

        Args:
            images: images with a bounding boxes

        Returns:
            detections
        Raises:
            LunaSDKException if an error occurs, context contains all errors
        """
        res = []
        errors = []
        errorDuringProgress = False
        for image in images:
            imageRes = []
            for bBox in image.bBoxes:
                try:
                    detection = self.redetectOne(image.image, bBox=bBox)
                    imageRes.append(detection)
                    errors.append(LunaVLError.Ok.format(LunaVLError.Ok.description))
                except LunaSDKException as exc:
                    errors.append(exc.error)
                    errorDuringProgress = True
                    break
            res.append(imageRes)
        if errorDuringProgress:
            raise LunaSDKException(LunaVLError.BatchedInternalError, errors)
        return res
