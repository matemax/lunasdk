from collections import namedtuple
from typing import List, Union, Type

import itertools

from lunavl.sdk.base import BoundingBox, LandmarkWithScore
from lunavl.sdk.detectors.base import BaseDetection
from lunavl.sdk.detectors.facedetector import FaceDetection, FaceDetector, Landmarks5, Landmarks68
from lunavl.sdk.detectors.humandetector import HumanDetection, HumanDetector, Landmarks17
from lunavl.sdk.faceengine.engine import DetectorType
from lunavl.sdk.image_utils.geometry import Point
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ONE_FACE, SEVERAL_FACES, SMALL_IMAGE, BAD_IMAGE

VLIMAGE_SMALL = VLImage.load(filename=SMALL_IMAGE)
VLIMAGE_ONE_FACE = VLImage.load(filename=ONE_FACE)
VLIMAGE_BAD_IMAGE = VLImage.load(filename=BAD_IMAGE)
VLIMAGE_SEVERAL_FACE = VLImage.load(filename=SEVERAL_FACES)
GOOD_AREA = Rect(100, 100, VLIMAGE_ONE_FACE.rect.width - 100, VLIMAGE_ONE_FACE.rect.height - 100)
OUTSIDE_AREA = Rect(100, 100, VLIMAGE_ONE_FACE.rect.width, VLIMAGE_ONE_FACE.rect.height)
AREA_WITHOUT_FACE = Rect(50, 50, 100, 100)
INVALID_RECT = Rect(0, 0, 0, 0)
ERROR_CORE_RECT = Rect(0.1, 0.1, 0.1, 0.1)  # anything out of range (0.1, 1)


class BaseDetectorTestClass(BaseTestClass):
    """
    Base class for detectors tests
    """

    #: detection class
    detectionClass: Type[BaseDetection]

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

    @staticmethod
    def assertPoint(point: Point):
        """
        Assert landmark point
        Args:
            point: point
        """
        assert isinstance(point, Point), "Landmarks does not contains Point"
        assert isinstance(point.x, float) and isinstance(point.y, float), "point coordinate is not float"

    def assertDetection(
        self,
        detection: Union[FaceDetection, HumanDetection, Union[List[FaceDetection], List[HumanDetection]]],
        imageVl: VLImage,
    ):
        """
        Function checks if an instance is Detection class, image and bounding box

        Args:
            detection: detection
            imageVl: class image
        """
        if isinstance(detection, list):
            detectionList = detection
        else:
            detectionList = [detection]  # type: ignore

        for detection in detectionList:
            assert isinstance(detection, self.__class__.detectionClass), (
                f"{detection.__class__} is not " f"{self.__class__.detectionClass}"
            )
            assert detection.image.asPillow() == imageVl.asPillow(), "Detection image does not match VLImage"
            self.assertBoundingBox(detection.boundingBox)


class HumanDetectTestClass(BaseDetectorTestClass):
    """
    Base class for human detection tests
    """

    #: global human detector
    detector: HumanDetector
    detectionClass: Type[HumanDetection] = HumanDetection

    @classmethod
    def setup_class(cls):
        """
        Create list of face detector
        """
        super().setup_class()
        cls.detector = cls.faceEngine.createHumanDetector()
        CaseLandmarks = namedtuple("CaseLandmarks", ("detectLandmarks"))
        cls.landmarksCases = [CaseLandmarks(True), CaseLandmarks(False)]

    def assertHumanDetection(self, detection: Union[HumanDetection, List[HumanDetection]], imageVl: VLImage):
        """
        Function checks if an instance is FaceDetection class

        Args:
            detection: face detection
            imageVl: class image
        """
        self.assertDetection(detection, imageVl)

    @staticmethod
    def assertDetectionLandmarks(detection: HumanDetection, landmarksIsExpected: bool = False):
        """
        Assert human detection landmarks
        Args:
            detection: detection
            landmarksIsExpected: landmarks is expected or not

        """
        if landmarksIsExpected:
            assert isinstance(
                detection.landmarks17, Landmarks17
            ), f"{detection.landmarks17.__class__} is not {Landmarks17}"
        else:
            assert detection.landmarks17 is None, detection.landmarks17

    @staticmethod
    def assertLandmarksPoints(landmarksPoints: tuple):
        """
        Assert landmarks points

        Args:
            landmarksPoints: tuple of landmarks points
        """
        assert isinstance(landmarksPoints, tuple), f"{landmarksPoints} points is not tuple"
        for point in landmarksPoints:
            assert isinstance(point, LandmarkWithScore), "Landmarks does not contains Point"
            BaseDetectorTestClass.assertPoint(point.point)


class FaceDetectTestClass(BaseDetectorTestClass):
    detectors: List[FaceDetector]
    detectionClass: Type[FaceDetection] = FaceDetection

    @classmethod
    def setup_class(cls):
        """
        Create list of face detector
        """
        super().setup_class()
        cls.detectors = [
            cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V1),
            cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V2),
            cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3),
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
        self.assertDetection(detection, imageVl)

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
            BaseDetectorTestClass.assertPoint(point)
