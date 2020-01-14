import itertools
from collections import namedtuple
from typing import List, Union, Tuple, ContextManager, Iterator, Type

from lunavl.sdk.faceengine.engine import VLFaceEngine, DetectorType
from lunavl.sdk.faceengine.facedetector import FaceDetection, FaceDetector, BoundingBox, Landmarks5, Landmarks68
from lunavl.sdk.image_utils.geometry import Point
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass

Detector: Type[Tuple[FaceDetector]] = namedtuple("Detector", ("type",))


class DetectTestClass(BaseTestClass):
    faceEngine: VLFaceEngine
    detectors: List[Detector]

    @classmethod
    def setup_class(cls):
        """
        Create list of face detector
        """
        super().setup_class()
        cls.detectors = [
            Detector(cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)),
            Detector(cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V2)),
            Detector(cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)),
        ]
        CaseLandmarks = namedtuple("CaseLandmarks", ("detect5Landmarks", "detect68Landmarks"))
        cls.landmarksCases = [
            CaseLandmarks(landmarks5, landmarks68)
            for landmarks5, landmarks68 in itertools.product((True, False), (True, False))
        ]

    def assertFaceDetection(self, detection: Union[FaceDetection, List[FaceDetection]], imageVl: VLImage):
        """
        Function checks if an instance is FaceDetection class

        Args:
            detection: face detection
            imageVl: class image
        """
        if isinstance(detection, list):
            listOfFaceDetection = [faceDetection for faceDetection in detection]
        else:
            listOfFaceDetection = [detection]

        for detection in listOfFaceDetection:
            assert isinstance(detection, FaceDetection), f"{detection.__class__} is not {FaceDetection}"
            assert detection.image == imageVl, "Detection image does not match VLImage"
            self.assertBoundingBox(detection.boundingBox)

    @staticmethod
    def assertDetectionLandmarks(
        detection: FaceDetection, landmarks5: Landmarks5 = None, landmarks68: Landmarks68 = None
    ):
        if landmarks5:
            assert isinstance(detection.landmarks5, Landmarks5), f"{detection.landmarks5.__class__} is not {Landmarks5}"
        else:
            assert detection.landmarks5 is None, detection.landmarks5
        if landmarks68:
            assert isinstance(
                detection.landmarks68, Landmarks68
            ), f"{detection.landmarks68.__class__} is not {Landmarks68}"
        else:
            assert detection.landmarks68 is None, detection.landmarks68

    @staticmethod
    def assertLandmarksPoints(landmarksPoints: tuple):
        """
        Assert landmarks points

        Args:
            landmarksPoints: tuple of landmarks points
        """
        assert isinstance(landmarksPoints, tuple), f"{landmarksPoints} points is not tuple"
        for point in landmarksPoints:
            assert isinstance(point, Point), "Landmarks does not contains Point"
            assert isinstance(point.x, float) and isinstance(point.y, float), "point coordinate is not float"

    def assertBoundingBox(self, boundingBox: BoundingBox):
        """
        Assert attributes of Bounding box class

        Args:
            boundingBox: bounding box
        """
        assert isinstance(boundingBox, BoundingBox), f"{boundingBox} is not {BoundingBox}"
        self.checkRectAttr(boundingBox.rect)

        assert isinstance(boundingBox.score, float), f"{boundingBox.score} is not float"
        assert 0 <= boundingBox.score < 1, "score out of range [0,1]"

    def detectorSubTest(self) -> Iterator[Tuple[ContextManager, FaceDetector]]:
        """
        Generator for sub tests from FaceDetector
        """
        for testDetector in self.detectors:
            subTest = self.subTest(testDetector=testDetector)
            detector = testDetector.type
            yield subTest, detector
