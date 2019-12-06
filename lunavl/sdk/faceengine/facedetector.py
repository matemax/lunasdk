"""
Module contains function for detection faces on images.
"""
from typing import Optional, Union, List, NamedTuple, Dict, Any, overload

from FaceEngine import DetectionFloat, FSDKError, dtBBox  # pylint: disable=E0611,E0401
from FaceEngine import DetectionType, Face  # pylint: disable=E0611,E0401
from FaceEngine import Landmarks5 as CoreLandmarks5  # pylint: disable=E0611,E0401
from FaceEngine import Landmarks68 as CoreLandmarks68  # pylint: disable=E0611,E0401
from FaceEngine import dt5Landmarks, dt68Landmarks  # pylint: disable=E0611,E0401
from lunavl.sdk.estimators.base_estimation import BaseEstimation

from ..errors.errors import LunaVLError
from ..errors.exceptions import LunaSDKException, CoreExceptionWrap
from ..image_utils.geometry import Rect, Landmarks
from ..image_utils.image import VLImage, ColorFormat


class ImageForDetection(NamedTuple):
    """
    Structure for the transfer to detector an image and detect an area.

    Attributes
        image (VLImage): image for detection
        detectArea (Rect[float]): area for face detection
    """

    image: VLImage
    detectArea: Rect


class ImageForRedetection(NamedTuple):
    """
    Structure for a redetector with an image and a area to detect in.

    Attributes
        image (VLImage): image for detection
        bBoxes (Rect): face bounding boxes
    """

    image: VLImage
    bBoxes: List[Rect]


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
            coreLandmarks5 = self.landmarks5.coreEstimation
            res["landmarks5"] = tuple((coreLandmarks5[index].x,
                                       coreLandmarks5[index].y) for index in range(5))
        if self.landmarks68 is not None:
            coreLandmarks68 = self.landmarks68.coreEstimation
            res["landmarks68"] = tuple((coreLandmarks68[index].x,
                                        coreLandmarks68[index].y) for index in range(68))
        return res


class FaceDetector:
    """
    Face detector.

    Attributes:
        _detector (IDetectorPtr): core detector
        detectorType (DetectionType): detector type

    """

    __slots__ = ("_detector", "detectorType")

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

    @CoreExceptionWrap(LunaVLError.DetectOneFaceError)
    def detectOne(
            self,
            image: VLImage,
            detectArea: Optional[Rect] = None,
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
            details = "Bad image format for detection, format: {}, image: {}".format(
                image.format.value, image.filename
            )
            raise LunaSDKException(LunaVLError.InvalidImageFormat.format(details))

        if detectArea is None:
            _detectArea = image.coreImage.getRect()
        else:
            _detectArea = detectArea.coreRectI

        error, detectRes = self._detector.detectOne(
            image.coreImage, _detectArea, self._getDetectionType(detect5Landmarks, detect68Landmarks)
        )
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        if not detectRes.isValid():
            return None
        coreDetection = detectRes
        return FaceDetection(coreDetection, image)

    @CoreExceptionWrap(LunaVLError.DetectFacesError)
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
                detectAreas.append(image.detectArea.coreRectI)
            if img.format != ColorFormat.R8G8B8:
                details = "Bad image format for detection, format: {}, image: {}".format(img.format.value, img.filename)
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

    @overload
    def redetectOne(
            self,
            image: VLImage,
            *,
            bBox: Optional[Rect],
            detect5Landmarks: bool = True,
            detect68Landmarks: bool = False,
    ) -> DetectionFloat:
        ...

    @overload
    def redetectOne(
            self,
            image: VLImage,
            *,
            detection: Optional[FaceDetection],
            detect5Landmarks: bool = True,
            detect68Landmarks: bool = False,
    ) -> DetectionFloat:
        ...

    @CoreExceptionWrap(LunaVLError.DetectFacesError)
    def redetectOne(self, image, *,
                    bBox: Optional[Rect] = None,
                    detection: Optional[FaceDetection] = None,
                    detect5Landmarks=True, detect68Landmarks=False) -> Union[None, FaceDetection]:
        """
        Redetect face on an image in area, restricted with image.bBox, bBox or detection.

        Args:
            image: image with a bounding box, or just VLImage. If VLImage provided, one of bBox or detection
                should be defined.
            bBox: bounding box
            detection: core detection
            detect5Landmarks: detect or not landmarks5
            detect68Landmarks: detect or not landmarks68

        Returns:
            detection if face found otherwise None
        Raises:
            LunaSDKException if an error occurs
        """
        if isinstance(image, VLImage) and (bBox is not None) and (detection is None):
            error, detectRes = self._detector.redetectOne(image.coreImage, bBox.coreRectF,
                                                          self._getDetectionType(detect5Landmarks, detect68Landmarks))
            vlImage = image
        elif isinstance(image, VLImage) and (bBox is None) and (detection is not None):
            error, detectRes = self._detector.redetectOne(image.coreImage, detection.coreEstimation.detection.rect,
                                                          self._getDetectionType(detect5Landmarks, detect68Landmarks))
            vlImage = image
        else:
            raise NotImplementedError

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        if detectRes.isValid():
            return FaceDetection(detectRes, vlImage)
        return None

    @CoreExceptionWrap(LunaVLError.DetectFacesError)
    def redetect(
            self,
            images: List[ImageForRedetection],
            detect5Landmarks: bool = True,
            detect68Landmarks: bool = False,
    ) -> List[List[Union[FaceDetection, None]]]:
        """
        Redetect face on each image.image in area, restricted with image.bBox.

        Args:
            images: images with a bounding boxes
            detect5Landmarks: detect or not landmarks5
            detect68Landmarks: detect or not landmarks68

        Returns:
            detections
        Raises:
            LunaSDKException if an error occurs
        """

        def facesFactory(image: ImageForRedetection) -> List[Face]:
            faces = [Face(image.image.coreImage, DetectionFloat(bBox.coreRectF, 1.0)) for bBox in image.bBoxes]
            return faces

        faces = []
        for image in images:
            faces.extend(facesFactory(image))
        error, detectRes = self._detector.redetect(faces, self._getDetectionType(detect5Landmarks, detect68Landmarks))
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

        detectIter = iter(detectRes)
        res = []
        for image in images:
            imageRes = []
            for _ in range(len(image.bBoxes)):
                detection = next(detectIter)
                if detection.isValid():
                    imageRes.append(FaceDetection(detection, image.image))
                else:
                    imageRes.append(None)
            res.append(imageRes)

        return res

    def setDetectionComparer(self):
        """
        todo: wtf
        Returns:

        """
        pass
