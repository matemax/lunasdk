"""
Module contains function for detection human bodies on images.
"""
from typing import Optional, Union, List, Dict, Any

from FaceEngine import HumanDetectionType, Human  # pylint: disable=E0611,E0401
from FaceEngine import HumanLandmarks17 as CoreLandmarks17  # pylint: disable=E0611,E0401

from .base import (
    ImageForDetection,
    ImageForRedetection,
    BaseDetection,
    assertImageForDetection,
    getArgsForCoreDetectorForImages,
)
from ..base import LandmarksWithScore
from ..errors.errors import LunaVLError, ErrorInfo
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
        human.detection.rect = image.bBoxes[index].coreRectF
        human.detection.score = 1
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
        toDetect = HumanDetectionType.DCT_BOX

        if detectLandmarks:
            toDetect = HumanDetectionType.DCT_ALL
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

        if detectArea is None:
            forDetection = ImageForDetection(image=image, detectArea=image.rect)
        else:
            forDetection = ImageForDetection(image=image, detectArea=detectArea)
        imgs, detectAreas = getArgsForCoreDetectorForImages([forDetection])
        error, detectRes = self._detector.detect(
            [imgs[0]], [detectAreas[0]], 1, self._getDetectionType(detectLandmarks)
        )
        assertError(error)

        return HumanDetection(detectRes[0][0], image) if detectRes[0] else None

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
        Raises:
            LunaSDKException(LunaVLError.InvalidImageFormat): if any image has bad format or detect is failed

        """
        imgs, detectAreas = getArgsForCoreDetectorForImages(images)
        detectionType = self._getDetectionType(detectLandmarks)

        error, detectRes = self._detector.detect(imgs, detectAreas, limit, detectionType)
        if error.isError:
            errors = []
            for image, detectArea in zip(imgs, detectAreas):
                errorOne, _ = self._detector.detect([image], [detectArea], 1, detectionType)
                if errorOne.isOk:
                    errors.append(LunaVLError.Ok.format(LunaVLError.Ok.description))
                else:
                    errors.append(LunaVLError.fromSDKError(errorOne))
            raise LunaSDKException(LunaVLError.BatchedInternalError, errors)

        res = []
        for numberImage, imageDetections in enumerate(detectRes):
            image_ = images[numberImage]
            image = image_ if isinstance(image_, VLImage) else image_.image
            res.append([HumanDetection(coreDetection, image) for coreDetection in imageDetections])
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
        if isinstance(bBox, Rect):
            area = bBox
        else:
            area = bBox.boundingBox.rect

        human = _createCoreHumans(ImageForRedetection(image, [area]))[0]
        error, detectRes = self._detector.redetectOne(human)

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
