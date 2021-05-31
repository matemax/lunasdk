"""
Module contains function for detection faces on images.
"""
from typing import Optional, Union, List, Dict, Any

from FaceEngine import (
    Detection,
    IFaceDetectionBatchPtr,
    DetectionType,
    Face,
    Landmarks5 as CoreLandmarks5,
    Landmarks68 as CoreLandmarks68,
    DT_LANDMARKS5,
    DT_LANDMARKS68,
    Image as CoreImage,
    FSDKError,
)  # pylint: disable=E0611,E0401

from ..base import Landmarks
from ..detectors.base import (
    ImageForDetection,
    ImageForRedetection,
    BaseDetection,
    getArgsForCoreDetectorForImages,
    getArgsForCoreRedetect,
    validateBatchDetectInput,
)
from ..errors.errors import LunaVLError
from ..errors.exceptions import CoreExceptionWrap, assertError, LunaSDKException
from ..image_utils.geometry import Rect
from ..image_utils.image import VLImage


def _createCoreFaces(image: ImageForRedetection) -> List[Face]:
    """
    Create core faces for redetection
    Args:
        image: image and bounding boxes for redetection
    Returns:
        Face object list. one object for one bbox
    """
    return [Face(image.image.coreImage, Detection(bBox.coreRectF, 1.0)) for bBox in image.bBoxes]


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


class FaceDetection(BaseDetection):
    """
    Attributes:
        landmarks5 (Optional[Landmarks5]): optional landmarks5
        landmarks68 (Optional[Landmarks68]): optional landmarks5
    """

    __slots__ = ("landmarks5", "landmarks68")

    def __init__(self, coreDetection: Face, image: VLImage):
        """
        Init.

        Args:
            coreDetection: core detection
        """
        super().__init__(coreDetection, image)

        if coreDetection.landmarks5_opt.isValid():
            self.landmarks5: Optional[Landmarks5] = Landmarks5(coreDetection.landmarks5_opt.value())
        else:
            self.landmarks5 = None

        if coreDetection.landmarks68_opt.isValid():
            self.landmarks68: Optional[Landmarks68] = Landmarks68(coreDetection.landmarks68_opt.value())
        else:
            self.landmarks68 = None

    def asDict(self) -> Dict[str, Any]:
        """
        Convert face detection to dict (json).

        Returns:
            dict. required keys: 'rect', 'score'. optional keys: 'landmarks5', 'landmarks68'
        """
        res = super().asDict()
        if self.landmarks5 is not None:
            coreLandmarks5 = self.landmarks5.coreEstimation
            res["landmarks5"] = tuple(
                (int(coreLandmarks5[index].x), int(coreLandmarks5[index].y)) for index in range(5)
            )
        if self.landmarks68 is not None:
            coreLandmarks68 = self.landmarks68.coreEstimation
            res["landmarks68"] = tuple(
                (int(coreLandmarks68[index].x), int(coreLandmarks68[index].y)) for index in range(68)
            )
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
    def collectDetectionsResult(
        fsdkDetectRes: IFaceDetectionBatchPtr,
        coreImages: List[CoreImage],
        images: Union[List[Union[VLImage, ImageForDetection]], List[ImageForRedetection]],
    ):
        """
        Collect detection results from core reply and prepare face detections
        Args:
            fsdkDetectRes: fsdk (re)detect results
            coreImages: core images
            images: incoming images
        Returns:
            return list of lists detection, order of detection lists is corresponding to order input images
        """
        res = []
        for imageIdx in range(fsdkDetectRes.getSize()):
            imagesDetections = []
            detections = fsdkDetectRes.getDetections(imageIdx)
            landmarks5Array = fsdkDetectRes.getLandmarks5(imageIdx)
            landmarks68Array = fsdkDetectRes.getLandmarks68(imageIdx)

            for detectionIdx, detection in enumerate(detections):
                face = Face(coreImages[imageIdx], detection)
                if landmarks5Array:
                    face.landmarks5_opt.set(landmarks5Array[detectionIdx])
                if landmarks68Array:
                    face.landmarks68_opt.set(landmarks68Array[detectionIdx])
                imagesDetections.append(face)

            image = images[imageIdx]
            vlImage = image if isinstance(image, VLImage) else image.image
            res.append(
                [
                    FaceDetection(coreDetection, vlImage) if coreDetection.isValid() else None
                    for coreDetection in imagesDetections
                ]
            )
        return res

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
            toDetect = toDetect | DT_LANDMARKS5
        if detect68Landmarks:
            toDetect = toDetect | DT_LANDMARKS68

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
            image: image. Format must be R8G8B8
            detectArea: rectangle area which contains face to detect. If not set will be set image.rect
            detect5Landmarks: detect or not landmarks5
            detect68Landmarks: detect or not landmarks68
        Returns:
            face detection if face is found otherwise None
        Raises:
            LunaSDKException: if detectOne is failed or image format has wrong  the format
        """

        if detectArea is None:
            _detectArea = image.coreImage.getRect()
        else:
            _detectArea = detectArea.coreRectI
        validateBatchDetectInput(self._detector, image.coreImage, _detectArea)
        error, detectRes = self._detector.detectOne(
            image.coreImage, _detectArea, self._getDetectionType(detect5Landmarks, detect68Landmarks)
        )
        assertError(error)

        if detectRes.isValid() is False:
            return None
        return FaceDetection(detectRes, image)

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
            LunaSDKException if an error occurs
        """
        coreImages, detectAreas = getArgsForCoreDetectorForImages(images)
        detectionType = self._getDetectionType(detect5Landmarks, detect68Landmarks)
        validateBatchDetectInput(self._detector, coreImages, detectAreas)
        error, fsdkDetectRes = self._detector.detect(coreImages, detectAreas, limit, detectionType)
        assertError(error)

        res = self.collectDetectionsResult(fsdkDetectRes, coreImages, images)
        return res

    @CoreExceptionWrap(LunaVLError.DetectFacesError)
    def redetectOne(  # noqa: F811
        self, image: VLImage, bBox: Union[Rect, FaceDetection], detect5Landmarks=True, detect68Landmarks=False
    ) -> Union[None, FaceDetection]:
        """
        Redetect face on an image in area, restricted with image.bBox, bBox or detection.

        Args:
            image: image with a bounding box, or just VLImage. If VLImage provided, one of bBox or detection
                should be defined.
            bBox: detection bounding box
            detect5Landmarks: detect or not landmarks5
            detect68Landmarks: detect or not landmarks68

        Returns:
            detection if face found otherwise None
        Raises:
            LunaSDKException if an error occurs
        """
        if isinstance(bBox, Rect):
            coreBBox = Detection(bBox.coreRectF, 1.0)
        else:
            coreBBox = bBox.coreEstimation.detection
        self._validateReDetectInput(image.coreImage, coreBBox)
        error, detectRes = self._detector.redetectOne(
            image.coreImage, coreBBox, self._getDetectionType(detect5Landmarks, detect68Landmarks)
        )
        assertError(error)

        if detectRes.isValid():
            return FaceDetection(detectRes, image)
        return None

    def _validateReDetectInput(self, coreImages: List[CoreImage], detectAreas: List[List[Detection]]):
        """
        Validate input data for face re-detect
        Args:
            coreImages:core images
            detectAreas: face re-detect areas
        Raises:
            LunaSDKException(LunaVLError.BatchedInternalError): if validation failed and coreImages has type list
                                                                                                  (batch redetect)
            LunaSDKException: if validation failed and coreImages has type CoreImage
        """
        if isinstance(coreImages, list):
            validationError, imagesErrors = self._detector.validate(coreImages, detectAreas)
        else:
            validationError, imagesErrors = self._detector.validate([coreImages], [[detectAreas]])
        if validationError.isOk:
            return
        if validationError.error != FSDKError.ValidationFailed:
            raise LunaSDKException(
                LunaVLError.ValidationFailed.format("Failed validation"),
                [LunaVLError.fromSDKError(errors[0]) for errors in imagesErrors],
            )
        if not isinstance(coreImages, list):
            raise LunaSDKException(LunaVLError.fromSDKError(imagesErrors[0][0]))
        errors = []

        for imageErrors in imagesErrors:
            for error in imageErrors:
                if error.isOk:
                    continue
                errors.append(LunaVLError.fromSDKError(error))
                break
            else:
                errors.append(LunaVLError.Ok.format(LunaVLError.Ok.description))
        raise LunaSDKException(
            LunaVLError.BatchedInternalError.format(LunaVLError.fromSDKError(validationError).detail), errors
        )

    @CoreExceptionWrap(LunaVLError.DetectFacesError)
    def redetect(
        self, images: List[ImageForRedetection], detect5Landmarks: bool = True, detect68Landmarks: bool = False
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
        detectionType = self._getDetectionType(detect5Landmarks, detect68Landmarks)

        coreImages, detectAreas = getArgsForCoreRedetect(images)
        self._validateReDetectInput(coreImages, detectAreas)
        error, fsdkDetectRes = self._detector.redetect(coreImages, detectAreas, detectionType)
        assertError(error)

        res = self.collectDetectionsResult(fsdkDetectRes, coreImages, images)
        return res

    def setDetectionComparer(self):
        """"""
        pass
