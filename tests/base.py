import unittest
from operator import attrgetter
from typing import Dict

import numpy as np
from PIL import Image
from _pytest._code import ExceptionInfo

from lunavl.sdk.errors.errors import ErrorInfo
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import ColorFormat, VLImage
from tests.resources import ONE_FACE


class BaseTestClass(unittest.TestCase):
    faceEngine: VLFaceEngine

    @classmethod
    def setup_class(cls):
        super().setUpClass()
        cls.faceEngine = VLFaceEngine()

    @staticmethod
    def assertLunaVlError(exceptionInfo: ExceptionInfo, expectedError: ErrorInfo):
        """
        Assert LunaVl Error

        Args:
            exceptionInfo: response from service
            expectedError: expected error
        """
        assert exceptionInfo.value.error.errorCode == expectedError.errorCode, exceptionInfo.value
        assert exceptionInfo.value.error.description == expectedError.description, exceptionInfo.value
        if expectedError.detail != "":
            assert exceptionInfo.value.error.detail == expectedError.detail, exceptionInfo.value

    @staticmethod
    def checkRectAttr(defaultRect: Rect):
        """
        Validate attributes of Rect

        Args:
            defaultRect: rect object
        """
        for rectType in ("coreRectI", "coreRectF"):
            assert all(
                isinstance(
                    getattr(defaultRect.__getattribute__(rectType), f"{coordinate}"),
                    float if rectType == "coreRectF" else int,
                )
                for coordinate in ("x", "y", "height", "width")
            )

    @staticmethod
    def getColorToImageMap() -> Dict[ColorFormat, VLImage]:
        """
        Get all available images.

        Returns:
            color format to vl image map
        """
        image = Image.open(ONE_FACE)
        R, G, B = np.array(image).T
        X = np.ndarray(B.shape, dtype=np.uint8)

        allImages = {
            ColorFormat.B8G8R8: VLImage.fromNumpyArray(np.array((B, G, R)).T, ColorFormat.B8G8R8),
            ColorFormat.B8G8R8X8: VLImage.fromNumpyArray(np.array((B, G, R, X)).T, ColorFormat.B8G8R8X8),
            ColorFormat.IR_X8X8X8: VLImage.fromNumpyArray(np.array(image, dtype=np.uint8), ColorFormat.IR_X8X8X8),
            ColorFormat.R16: VLImage.fromNumpyArray(np.array(image.convert("L"), dtype=np.uint16), ColorFormat.R16),
            ColorFormat.R8: VLImage.fromNumpyArray(np.array(image.convert("L"), dtype=np.uint8).T, ColorFormat.R8),
            ColorFormat.R8G8B8: VLImage.fromNumpyArray(np.array(image), ColorFormat.R8G8B8),
            ColorFormat.R8G8B8X8: VLImage.fromNumpyArray(np.array((R, G, B, X)).T, ColorFormat.R8G8B8X8),
        }
        notImplementedFormats = set(ColorFormat) - set(allImages) - {ColorFormat.Unknown}
        if notImplementedFormats:
            notImplementedFormatsList = list(map(attrgetter("name"), notImplementedFormats))
            raise RuntimeError(f"Add Image for {notImplementedFormatsList} color formats")
        return allImages
