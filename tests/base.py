import unittest

from _pytest._code import ExceptionInfo

from lunavl.sdk.errors.errors import ErrorInfo
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.geometry import Rect


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
