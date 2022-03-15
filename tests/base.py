import unittest
from operator import attrgetter
from typing import Dict, Any, Type

import numpy as np
from PIL import Image
from _pytest._code import ExceptionInfo

from lunavl.sdk.async_task import AsyncTask
from lunavl.sdk.errors.errors import ErrorInfo
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import ColorFormat, VLImage
from tests.resources import ONE_FACE


class BaseTestClass(unittest.TestCase):
    faceEngine = VLFaceEngine()

    @classmethod
    def setup_class(cls):
        super().setUpClass()

    @classmethod
    def teardown_class(cls) -> None:
        super().tearDownClass()

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
    def assertReceivedAndRawExpectedErrors(receivedError: ErrorInfo, expectedErrorEmptyDetail: ErrorInfo):
        """
        Assert expected and received errors as dicts
        Args:
            receivedError: received error
            expectedErrorEmptyDetail: expected error with empty detail
        """
        assert expectedErrorEmptyDetail.errorCode == receivedError.errorCode
        assert expectedErrorEmptyDetail.description == receivedError.description
        assert expectedErrorEmptyDetail.description == receivedError.detail

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
    def generateColorToArrayMap() -> Dict[ColorFormat, np.ndarray]:
        """
        Get images as ndarrays in all available color formats.

        Returns:
            color format to pixel ndarray map
        """
        image = Image.open(ONE_FACE)
        R, G, B = np.array(image).T
        X = np.ndarray(B.shape, dtype=np.uint8)

        allImages = {
            ColorFormat.B8G8R8: np.array((B, G, R)).T,
            ColorFormat.B8G8R8X8: np.array((B, G, R, X)).T,
            ColorFormat.IR_X8X8X8: np.array(image, dtype=np.uint8),
            ColorFormat.R16: np.array(image.convert("L"), dtype=np.uint16),
            ColorFormat.R8: np.array(image.convert("L"), dtype=np.uint8),
            ColorFormat.R8G8B8: np.array(image),
            ColorFormat.R8G8B8X8: np.array((R, G, B, X)).T,
        }

        def _checksAllFormats():
            _notImplementedFormats = set(ColorFormat) - set(allImages) - {ColorFormat.Unknown}
            if _notImplementedFormats:
                notImplementedFormatsList = list(map(attrgetter("name"), _notImplementedFormats))
                raise RuntimeError(f"Add Image for {notImplementedFormatsList} color formats")

        def _checksArrayShapes():
            for color, ndarray in allImages.items():
                if ndarray.shape[:2] != allImages[ColorFormat.R8G8B8].shape[:2]:
                    msg = (
                        f"'{color.name}' image has incorrect shape.\n"
                        f"Expected:{allImages[ColorFormat.R8G8B8].shape}\n"
                        f"Received:{ndarray.shape}"
                    )
                    raise RuntimeError(msg)

        _checksAllFormats()
        _checksArrayShapes()
        return allImages

    @staticmethod
    def getColorToImageMap() -> Dict[ColorFormat, VLImage]:
        """
        Get images as vl image in all available color formats.

        Returns:
            color format to vl image map
        """
        return {
            color: VLImage.fromNumpyArray(ndarray, color)
            for color, ndarray in BaseTestClass.generateColorToArrayMap().items()
        }

    @staticmethod
    def assertAsyncEstimation(task: AsyncTask, expectedTypeResult: Type[Any]):
        """
        Assert single async estimation
        Args:
            task: async task
            expectedTypeResult: expected type of result
        """
        isinstance(task, AsyncTask)
        res = task.get()
        isinstance(res, expectedTypeResult)

    @staticmethod
    def assertAsyncBatchEstimation(task: AsyncTask, expectedTypeResult: Type[Any]):
        """
        Assert batch async estimation
        Args:
            task: async task
            expectedTypeResult: expected type of result
        """
        isinstance(task, AsyncTask)
        res = task.get()
        isinstance(res, list)
        isinstance(res[0], expectedTypeResult)

    @staticmethod
    def assertAsyncBatchEstimationWithAggregation(task: AsyncTask, expectedTypeResult: Type[Any]):
        """
        Assert batch async estimation with aggregation
        Args:
            task: async task
            expectedTypeResult: expected type of result
        """
        isinstance(task, AsyncTask)
        res = task.get()
        isinstance(res, tuple)
        assert 2 == len(res)
        isinstance(res[0][0], expectedTypeResult)
        isinstance(res[1], expectedTypeResult)
