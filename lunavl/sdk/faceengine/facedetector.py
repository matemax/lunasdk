"""
Module contains function for detection faces on images.
"""
from typing import Optional, Union, List, NamedTuple, Dict, Any

from FaceEngine import DetectionFloat, FSDKError  # pylint: disable=E0611,E0401
from FaceEngine import DetectionType, Face  # pylint: disable=E0611,E0401
from FaceEngine import Landmarks5 as CoreLandmarks5  # pylint: disable=E0611,E0401
from FaceEngine import Landmarks68 as CoreLandmarks68  # pylint: disable=E0611,E0401
from FaceEngine import dt5Landmarks, dt68Landmarks  # pylint: disable=E0611,E0401
from lunavl.sdk.estimators.base_estimation import BaseEstimation

from ..errors.errors import LunaVLError
from ..errors.exceptions import LunaSDKException, CoreExceptionWarp
from ..image_utils.geometry import Rect, Landmarks
from ..image_utils.image import VLImage, ColorFormat


class ImageForDetection(NamedTuple):
    """
    Structure for the transfer to detector an image and detect an area.

    Attributes
        image (VLImage): image for detection
        detectArea (Rect[float]):
    """

    image: VLImage
    detectArea: Rect[float]


class Landmarks5(Landmarks):
    """
    Landmarks5
    """

    #  pylint: disable=W0235
    def __init__(self, coreLandmark5: CoreLandmarks5):
        """
        Init

        Args:
            coreLandmark5: core landmarks
        """
        super().__init__(coreLandmark5)


class Landmarks68(Landmarks):
    """
    Landmarks68
    """

    #  pylint: disable=W0235
    def __init__(self, coreLandmark68: CoreLandmarks68):
        """
        Init

        Args:
            coreLandmark68: core landmarks
        """
        super().__init__(coreLandmark68)


class BoundingBox(BaseEstimation):
    """
    Detection bounding box, it is characterized of rect and score:

        - rect (Rect[float]): face bounding box
        - score (float): face score (0,1), detection score is the measure of classification confidence
                         and not the source image quality. It may be used topick the most "*confident*" face of many.
    """

    #  pylint: disable=W0235
    def __init__(self, boundingBox: DetectionFloat):
        """
        Init.

        Args:
            boundingBox: core bounding box
        """
        super().__init__(boundingBox)

    @property
    def score(self) -> float:
        """
        Get score

        Returns:
            number in range [0,1]
        """
        return self._coreEstimation.score

    @property
    def rect(self) -> Rect[float]:
        """
        Get rect.

        Returns:
            float rect
        """
        return Rect.fromCoreRect(self._coreEstimation.rect)

    def asDict(self) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Convert to  dict.

        Returns:
            {"rect": self.rect, "score": self.score}
        """
        return {"rect": self.rect.asDict(), "score": self.score}


class FaceDetection(BaseEstimation):
    """
    Attributes:
        boundingBox (BoundingBox): face bounding box
        landmarks5 (Optional[Landmarks5]): optional landmarks5
        landmarks68 (Optional[Landmarks68]): optional landmarks5
        _image (VLImage): source of detection

    """

    __slots__ = (
        "boundingBox",
        "landmarks5",
        "landmarks68",
        "_coreDetection",
        "_image",
        "_emotions",
        "_quality",
        "_mouthState",
    )

    def __init__(self, coreDetection: Face, image: VLImage):
        """
        Init.

        Args:
            coreDetection: core detection
        """
        super().__init__(coreDetection)

        self.boundingBox = BoundingBox(coreDetection.detection)
        if coreDetection.landmarks5_opt.isValid():
            self.landmarks5: Optional[Landmarks5] = Landmarks5(coreDetection.landmarks5_opt.value())
        else:
            self.landmarks5 = None

        if coreDetection.landmarks68_opt.isValid():
            self.landmarks68: Optional[Landmarks68] = Landmarks68(coreDetection.landmarks68_opt.value())
        else:
            self.landmarks68 = None
        self._image = image
        self._emotions = None
        self._quality = None
        self._mouthState = None

    @property
    def image(self) -> VLImage:
        """
        Get source of detection.

        Returns:
            source image
        """
        return self._image

    def asDict(self) -> Dict[str, Any]:
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
        # TODO: may be nullable landmarks5?
        return res


class FaceDetector:
    """
    Face detector.

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
            toDetect = toDetect | dt5Landmarks
        if detect68Landmarks:
            toDetect = toDetect | dt68Landmarks

        return DetectionType(toDetect)

    @CoreExceptionWarp(LunaVLError.DetectOneFaceError)
    def detectOne(
        self,
        image: VLImage,
        detectArea: Optional[Rect[float]] = None,
        detect5Landmarks: bool = True,
        detect68Landmarks: bool = False,
    ) -> Union[None, FaceDetection]:
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
            LunaSDKException: if detectOne is failed or image format has wrong  the format
        """
        if image.format != ColorFormat.R8G8B8:
            details = "Bad image format for detection,  format: {}, image: {}".format(
                image.format.value, image.filename
            )
            raise LunaSDKException(LunaVLError.InvalidImageFormat.format(details))

        if detectArea is None:
            _detectArea = image.coreImage.getRect()
        else:
            _detectArea = detectArea.coreRect

        error, detectRes = self._detector.detectOne(
            image.coreImage, _detectArea, self._getDetectionType(detect5Landmarks, detect68Landmarks)
        )
        if error.isError:
            if error.FSDKError == FSDKError.BufferIsEmpty:
                return None
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        coreDetection = detectRes
        return FaceDetection(coreDetection, image)

    @CoreExceptionWarp(LunaVLError.DetectFacesError)
    def detect(
        self,
        images: List[Union[VLImage, ImageForDetection]],
        limit: int = 5,
        detect5Landmarks: bool = True,
        detect68Landmarks: bool = False,
    ) -> List[List[FaceDetection]]:
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
            LunaSDKException(LunaVLError.InvalidImageFormat): if any image has bad format or detect is failed

        """
        imgs = []
        detectAreas = []
        for image in images:

            if isinstance(image, VLImage):
                img = image
                detectAreas.append(image.coreImage.getRect())
            else:
                img = image.image
                detectAreas.append(image.detectArea.coreRect)
            if img.format != ColorFormat.R8G8B8:
                details = "Bad image format for detection, format {}, img {}".format(img.format.value, img.filename)
                raise LunaSDKException(LunaVLError.InvalidImageFormat.format(details))
            imgs.append(img.coreImage)

        error, detectRes = self._detector.detect(
            imgs, detectAreas, limit, self._getDetectionType(detect5Landmarks, detect68Landmarks)
        )
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        res = []
        for numberImage, imageDetections in enumerate(detectRes):
            image_ = images[numberImage]
            image = image_ if isinstance(image_, VLImage) else image_.image
            res.append([FaceDetection(coreDetection, image) for coreDetection in imageDetections])

        return res

    def redetectOne(self):
        """
        todo: wtf
        Returns:

        """
        pass

    def redect(self):
        """
        todo: wtf
        Returns:

        """
        pass

    def setDetectionComparer(self):
        """
        todo: wtf
        Returns:

        """
        pass
