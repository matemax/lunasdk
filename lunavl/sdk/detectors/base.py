from typing import NamedTuple, List, Any, Dict, Union, Tuple

from FaceEngine import Rect as CoreRectI, Detection, Image as CoreImage, FSDKError  # pylint: disable=E0611,E0401

from ..base import BaseEstimation, BoundingBox
from ..errors.errors import LunaVLError
from ..errors.exceptions import LunaSDKException
from ..image_utils.geometry import Rect
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


class BaseDetection(BaseEstimation):
    """
    Attributes:
        boundingBox (sdk.detectors.base.BoundingBox): face bounding box
        _image (VLImage): source of detection (may differ from the original image due to the orientation mode)

    """

    __slots__ = ("boundingBox", "_coreDetection", "_image")

    def __init__(self, coreDetection: Any, image: VLImage):
        """
        Init.

        Args:
            coreDetection: core detection
            image: original image
        """
        super().__init__(coreDetection)

        self.boundingBox = BoundingBox(coreDetection.detection)
        self._image = VLImage(body=coreDetection.img, filename=image.filename)

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
            dict. required keys: 'rect', 'score'.
        """
        return self.boundingBox.asDict()


def assertImageForDetection(image: VLImage) -> None:
    """
    Assert image for detection
    Args:
        image: image

    Raises:
        LunaSDKException: if image format is not R8G8B8
    """
    if image.format != ColorFormat.R8G8B8:
        details = "Bad image format for detection, format: {}, image: {}".format(image.format.value, image.filename)
        raise LunaSDKException(LunaVLError.InvalidImageFormat.format(details))


def getArgsForCoreDetectorForImages(
    images: List[Union[VLImage, ImageForDetection]]
) -> Tuple[List[CoreImage], List[CoreRectI]]:
    """
    Create args for detect for image list
    Args:
        images: list of images for detection

    Returns:
        tuple: first - list core images
               second - detect area for corresponding images
    """

    coreImages, detectAreas = [], []

    for image in images:
        if isinstance(image, VLImage):
            img = image
            detectAreas.append(image.coreImage.getRect())
        else:
            img = image.image
            detectAreas.append(image.detectArea.coreRectI)
        coreImages.append(img.coreImage)

    return coreImages, detectAreas


def getArgsForCoreRedetect(images: List[ImageForRedetection]) -> Tuple[List[CoreImage], List[List[Detection]]]:
    """
    Create args for redetect for image list
    Args:
        images: list of images for redetection

    Returns:
        tuple: first - list core images
               second - list detect area for corresponding images
    """
    coreImages, detectAreas = [], []

    for image in images:
        coreImages.append(image.image.coreImage)
        detectAreas.append([Detection(bbox.coreRect, 1.0) for bbox in image.bBoxes])

    return coreImages, detectAreas


def validateBatchDetectInput(
    detector, coreImages: Union[List[CoreImage]], detectAreas: Union[CoreRectI, List[CoreRectI]]
) -> None:
    """
    Collect errors from single operations and raise complex exception
    Args:
        detector: core face or body detector
        coreImages: list of core images
        detectAreas: list of detect areas for core images
    Raises:
        LunaSDKException: if validation are failed or data is not valid
    """
    limit = 1
    if not isinstance(coreImages, list):
        mainError, imageErrors = detector.validate([coreImages], [detectAreas], limit)
    else:
        mainError, imageErrors = detector.validate(coreImages, detectAreas, limit)
    if mainError.isOk:
        return
    if mainError.error != FSDKError.ValidationFailed:
        raise LunaSDKException(
            LunaVLError.ValidationFailed.format(mainError.what),
            [LunaVLError.fromSDKError(errors[0]) for errors in imageErrors],
        )
    if not isinstance(coreImages, list):
        raise LunaSDKException(LunaVLError.fromSDKError(imageErrors[0]))
    errors = []
    for error in imageErrors:
        if error.isOk:
            continue
        errors.append(LunaVLError.fromSDKError(error))
        break
    else:
        errors.append(LunaVLError.Ok.format(LunaVLError.Ok.description))
    raise LunaSDKException(
        LunaVLError.BatchedInternalError.format(LunaVLError.fromSDKError(mainError).detail), errors
    )
