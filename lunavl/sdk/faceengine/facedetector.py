from enum import Enum
from typing import Optional, Union, List, NamedTuple, Dict

import FaceEngine

from FaceEngine import ObjectDetectorClassType, DetectionType, Face, Landmarks5 as CoreLandmarks5, \
    Landmarks68 as CoreLandmarks68, DetectionFloat, FSDKError

from lunavl.sdk.errors.errors import Error
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.image_utils.geometry import Rect, Point
from lunavl.sdk.image_utils.image import VLImage, Format


class ImageForDetection(NamedTuple):
    """
    Structure for the transfer to detector an image and detect an area.

    Attributes
        image (VLImage): image for detection
        detectArea (Rect[float]):
    """
    image: VLImage
    detectArea: Rect[float]


class DetectorType(Enum):
    """
    Detector types enum
    """
    FACE_DET_DEFAULT = "FACE_DET_DEFAULT"  #: what is default?
    FACE_DET_V1 = "FACE_DET_V1"     #: todo description
    FACE_DET_V2 = "FACE_DET_V2"
    FACE_DET_V3 = "FACE_DET_V3"
    FACE_DET_COUNT = "FACE_DET_COUNT"

    @property
    def coreDetectorType(self) -> ObjectDetectorClassType:
        """
        Convert  self to core detector type

        Returns:
            ObjectDetectorClassType
        """
        return getattr(ObjectDetectorClassType, self.value)


class Landmarks5:
    """
    Landmarks5

    Attributes:
        points (List[Point[float]]): 5 point (todo reference)
    """
    def __init__(self, coreLandmark5: CoreLandmarks5):
        """
        Init

        Args:
            coreLandmark5: core landmarks
        """
        self.points = [Point.fromVector2(point) for point in coreLandmark5]


class Landmarks68:
    """
    Landmarks68

    Attributes:
        points (List[Point[float]]): 68 point (todo reference)
    """
    def __init__(self, coreLandmark68: CoreLandmarks68):
        """
        Init

        Args:
            coreLandmark68: core landmarks
        """
        self.points = [Point.fromVector2(point) for point in coreLandmark68]


class BoundingBox:
    """
    Attributes:
        rect (Rect[float]): face bounding box
        score (float): face score (0,1)
    """
    def __init__(self, boundingBox: DetectionFloat):
        """
        Init.

        Args:
            boundingBox: core bounding box
        """
        self.score = boundingBox.score
        self.rect = Rect.fromCoreRect(boundingBox.rect)


class FaceDetection:
    """
    Attributes:
        boundingBox (BoundingBox): face bounding box
        landmarks5 (Optional[Landmarks5]): optional landmarks5
        landmarks68 (Optional[Landmarks68]): optional landmarks5
    """
    __slots__ = ["boundingBox", "landmarks5", "landmarks68"]

    def __init__(self, coreDetection: Face):
        """
        Init.

        Args:
            coreDetection: core detection
        """
        self.boundingBox = BoundingBox(coreDetection.detection)
        if coreDetection.landmarks5_opt.isValid():
            self.landmarks5 = Landmarks5(coreDetection.landmarks5_opt.value())
        else:
            self.landmarks5 = None

        if coreDetection.landmarks68_opt.isValid():
            self.landmarks68 = Landmarks68(coreDetection.landmarks68_opt.value())
        else:
            self.landmarks68 = None

    def asDict(self)  -> Dict[str]:
        """
        Convert face detection to dict (json).

        Returns:
            dict. required keys: 'rect', 'score'. optional keys: 'landmarks5', 'landmarks68'
        """
        res = {"rect": self.boundingBox.rect.asDict(), "score": self.boundingBox.score}
        if self.landmarks5 is not None:
            res["landmarks5"] = [point.asDict() for point in self.landmarks5.points]
        if self.landmarks68 is not None:
            res["landmarks68"] = [point.asDict() for point in self.landmarks68.points]
        # todo: may be nullable landmarks5?
        return res


class FaceDetector:
    """
    Class contain
    Attributes:
        _detector (IDetectorPtr): core detector

    """
    __slots__ = ["_detector", "detectorType"]

    def __init__(self, detectorPtr, detectorType: DetectionType):
        self._detector = detectorPtr
        self.detectorType = detectorType

    @staticmethod
    def _getDetectionType(detect5Landmarks: bool = True, detect68Landmarks: bool = False) -> DetectionType:
        """
        Get  core detection type

        Args:
            detect5Landmarks: detect or not landmarks5
            detect68Landmarks: detect or not landmarks68

        Returns:
            detection type
        """
        toDetect = 0

        if detect5Landmarks:
            toDetect = toDetect | FaceEngine.dt5Landmarks
        if detect68Landmarks:
            toDetect = toDetect | FaceEngine.dt68Landmarks

        return DetectionType(toDetect)

    def detectOne(self, image: VLImage, detectArea: Optional[Rect[float]] = None, detect5Landmarks: bool = True,
                  detect68Landmarks: bool = False) -> Union[None, FaceDetection]:
        """
        Detect just one best detection on the image.

        Args:
            image: image. Format must be R8G8B8 (todo check)
            detectArea: rectangle area which contains face to detect. If not set will be set image.rect
            detect5Landmarks: detect or not landmarks5
            detect68Landmarks: detect or not landmarks68
        Returns:
            face detection if face is found otherwise None
        Raises:
            LunaSDKException: if detectOne is failed
        """
        if detectArea is None:
            _detectArea = image.coreImage.getRect()
        else:
            _detectArea = detectArea.coreRect

        detectRes = self._detector.detectOne(image.coreImage, _detectArea,
                                             self._getDetectionType(detect5Landmarks, detect68Landmarks))
        if detectRes[0].isError:
            if detectRes[0].FSDKError == FSDKError.BufferIsEmpty:
                return None
            error = Error.fromSDKError(123, "detection", detectRes[0])
            raise LunaSDKException(error)
        coreDetection = detectRes[1]
        if not coreDetection.detection.isValid():
            raise ValueError("WTF bad rect")  # todo check
        return FaceDetection(coreDetection)

    def detect(self, images: List[Union[VLImage, ImageForDetection]], limit: int = 5, detect5Landmarks: bool = True,
               detect68Landmarks: bool = False) -> List[List[FaceDetection]]:
        """
        Batch detect faces on images.

        Args:
            images: input images list. Format must be R8G8B8
            limit: max number of detections per input image
            detect5Landmarks: detect or not landmarks5
            detect68Landmarks: detect or not landmarks68
        Returns:
            return list of lists detection, order of detection lists is corresponding to order input images
        Raises:
            LunaSDKException: if any image has bad format or detect is failed

        """
        imgs = []
        detectAreas = []
        for image in images:

            if isinstance(image, VLImage):
                img = image
                detectArea = image.coreImage.getRect()
            else:
                img = image[0]
                detectArea = image[1].coreRect
            if img.format == Format.R8G8B8:
                error = Error(126, "bad format",
                              "Bad image format for detection {}, img {}".format(img.format.value, img.format))
                raise LunaSDKException(error)
            imgs.append(img.coreImage)
            detectAreas.append(detectArea)

        detectRes = self._detector.detect(imgs, detectAreas, limit,
                                          self._getDetectionType(detect5Landmarks, detect68Landmarks))
        if detectRes[0].isError:
            error = Error.fromSDKError(124, "detection", detectRes[0])
            raise LunaSDKException(error)
        res = []
        for imageDetections in detectRes[1]:
            res.append([FaceDetection(coreDetection) for coreDetection in imageDetections])
        return res