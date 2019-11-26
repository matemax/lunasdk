import unittest

from _pytest._code import ExceptionInfo

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.errors.errors import ErrorInfo


class BaseTestClass(unittest.TestCase):
    faceEngine: VLFaceEngine = None

    @classmethod
    def setup_class(cls):
        cls.faceEngine = VLFaceEngine()

    def assertLunaVlError(self, exceptionInfo: ExceptionInfo, expectedStatusCode: int, expectedError: ErrorInfo):
        """
        Assert LunaVl Error

        Args:
            exceptionInfo: response from service
            expectedStatusCode: expected status code
            expectedError: expected error
        """
        assert expectedStatusCode == exceptionInfo.value.error.errorCode, exceptionInfo.value
        assert expectedStatusCode == expectedError.errorCode, exceptionInfo.value
        assert exceptionInfo.value.error.description == expectedError.description, exceptionInfo.value
        if expectedError.detail != "":
            assert exceptionInfo.value.error.detail == expectedError.detail, exceptionInfo.value
