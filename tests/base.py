import unittest
from collections import namedtuple
from typing import List, Union, Generator, Tuple
from unittest.case import _SubTest

from _pytest._code import ExceptionInfo

from lunavl.sdk.errors.errors import ErrorInfo
from lunavl.sdk.faceengine.engine import VLFaceEngine, DetectorType
from lunavl.sdk.faceengine.facedetector import FaceDetection, FaceDetector
from lunavl.sdk.image_utils.geometry import Point, Rect
from lunavl.sdk.image_utils.image import VLImage


class BaseTestClass(unittest.TestCase):
    faceEngine: VLFaceEngine = None

    @classmethod
    def setup_class(cls):
        cls.faceEngine = VLFaceEngine()
        faceDetector = namedtuple("faceDetector", ("detector", "score"))
        cls.Detectors = [
            faceDetector(cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V1), 0.9983),
            faceDetector(cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V2), 0.9913),
            faceDetector(cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3), 0.9999),
        ]

    @staticmethod
    def assertLunaVlError(exceptionInfo: ExceptionInfo, expectedStatusCode: int, expectedError: ErrorInfo):
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

    @staticmethod
    def assertDetectionTrue(detection: Union[FaceDetection, List[List[FaceDetection]]], imageVl: List[VLImage]):
        """
        Assert FaceDetection is true

        Args:
            detection: face detection
            imageVl: class image
        """
        if isinstance(detection, list):
            listOfFaceDetection = [faceDetection for item in detection for faceDetection in item]
        else:
            listOfFaceDetection = [detection]

        for detection in listOfFaceDetection:
            assert isinstance(detection, FaceDetection), detection
            assert detection.image in imageVl, "Detection image does not match VLImage"
            assert detection.boundingBox.rect.isValid()

    @staticmethod
    def assertLandmarksPoints(landmarksPoints: tuple):
        """
        Assert landmarks points
        Args:
            landmarksPoints: tuple of landmarks points
        """
        assert isinstance(landmarksPoints, tuple), "Landmarks points is not tuple"
        for point in landmarksPoints:
            assert isinstance(point, Point), "Landmarks does not contains Point"
            assert isinstance(point.x, float) and isinstance(point.y, float), "point coordinate is not float"

    @staticmethod
    def checkRectAttr(defaultRect: Rect, isImage: bool = False):
        for rectType in ("coreRectI", "coreRectF"):
            assert all(isinstance(getattr(defaultRect.__getattribute__(rectType), f"{coordinate}"),
                                  float if rectType == "coreRectF" else int)
                       for coordinate in ("x", "y", "height", "width"))
        assert all(isinstance(getattr(defaultRect, f"{x}"), int if isImage else float)
                   for x in ("x", "y", "height", "width"))

    def detectorSubTest(self) -> Generator[None, Tuple[_SubTest, FaceDetector, float], None]:
        """
        Generator for sub tests from FaceDetector
        """
        for testDetector in self.Detectors:
            subTest = self.subTest(testDetector=testDetector)
            detector = testDetector.detector
            score = testDetector.score
            yield subTest, detector, score
