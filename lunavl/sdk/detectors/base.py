from typing import NamedTuple, List, Any, Dict, Union

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException

from ..base import BaseEstimation, BoundingBox
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
        _image (VLImage): source of detection

    """

    __slots__ = ("boundingBox", "_coreDetection", "_image")

    def __init__(self, coreDetection: Any, image: VLImage):
        """
        Init.

        Args:
            coreDetection: core detection
        """
        super().__init__(coreDetection)

        self.boundingBox = BoundingBox(coreDetection.detection)
        self._image = image

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
        return {"rect": self.boundingBox.rect.asDict(), "score": self.boundingBox.score}


def assertImageForDetection(image: VLImage):
    if image.format != ColorFormat.R8G8B8:
        details = "Bad image format for detection, format: {}, image: {}".format(image.format.value, image.filename)
        raise LunaSDKException(LunaVLError.InvalidImageFormat.format(details))


def getDataForCoreDetector(images: List[Union[VLImage, ImageForDetection]]):
    coreImages = []
    detectAreas = []
    for image in images:

        if isinstance(image, VLImage):
            img = image
            detectAreas.append(image.coreImage.getRect())
            assertImageForDetection(image)
        else:
            img = image.image
            detectAreas.append(image.detectArea.coreRectI)
            assertImageForDetection(image.image)
        coreImages.append(img.coreImage)
    return coreImages, detectAreas
