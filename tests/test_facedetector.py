import itertools
import json
from collections import namedtuple
from typing import Tuple, Union, List, Dict

import jsonschema
import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.faceengine.facedetector import (
    FaceDetector,
    Landmarks5,
    Landmarks68,
    FaceDetection,
    ImageForDetection, BoundingBox)
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage, ColorFormat
from tests.base import BaseTestClass
from tests.resources import ONE_FACE, SEVERAL_FACES, MANY_FACES, NO_FACES
from tests.schemas import REQUIRED_FACE_DETECTION, LANDMARKS5, LANDMARKS68

VLIMAGE_ONE_FACE = VLImage.load(filename=ONE_FACE)
VLIMAGE_SEVERAL_FACE = VLImage.load(filename=SEVERAL_FACES)
GOOD_AREA = Rect(100, 100, VLIMAGE_ONE_FACE.rect.width - 100, VLIMAGE_ONE_FACE.rect.height - 100)


class TestDetector(BaseTestClass):
    """
    Test of detector.
    """

    detector: FaceDetector = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)

    @staticmethod
    def getDetections(detect5Landmarks: bool = True,
                      detect68Landmarks: bool = False) -> Tuple[FaceDetection, FaceDetection]:
        """
        Function for detection with default face detector

        Returns:
            FaceDetection
        """
        detectOne = TestDetector.detector.detectOne(image=VLIMAGE_ONE_FACE, detect5Landmarks=detect5Landmarks,
                                                    detect68Landmarks=detect68Landmarks)
        detect = TestDetector.detector.detect(images=[VLIMAGE_ONE_FACE], detect5Landmarks=detect5Landmarks,
                                              detect68Landmarks=detect68Landmarks)[0][0]
        return detectOne, detect

    def test_image_for_detection(self):
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                detection = detector.detect(images=[ImageForDetection(VLIMAGE_ONE_FACE, GOOD_AREA)])
                self.assertDetectionTrue(detection, [VLIMAGE_ONE_FACE])
                assert 1 == len(detection)

    def test_check_landmarks_points(self):
        for detection in self.getDetections(detect68Landmarks=True):
            self.assertDetectionTrue(detection, [VLIMAGE_ONE_FACE])

            self.assertLandmarksPoints(detection.landmarks5.points)
            self.assertLandmarksPoints(detection.landmarks68.points)
            assert 5 == len(detection.landmarks5.points)
            assert 68 == len(detection.landmarks68.points)

    def test_landmarks_as_dict(self):
        for detection in self.getDetections(detect68Landmarks=True):
            self.assertDetectionTrue(detection, [VLIMAGE_ONE_FACE])

            currentLandmarks5 = json.loads(json.dumps(detection.landmarks5.asDict()))
            assert jsonschema.validate(currentLandmarks5, LANDMARKS5) is None, currentLandmarks5

            currentLandmarks68 = json.loads(json.dumps(detection.landmarks68.asDict()))
            assert jsonschema.validate(currentLandmarks68, LANDMARKS68) is None, currentLandmarks68

    def test_valid_bounding_box(self):
        for detection in self.getDetections(detect68Landmarks=True):
            self.assertDetectionTrue(detection, [VLIMAGE_ONE_FACE])
            rectBBox = detection.boundingBox.rect
            assert rectBBox.isValid(), "Invalid width and height"
            assert all((isinstance(rectBBox.x, float), isinstance(rectBBox.y, float),
                        isinstance(rectBBox.width, float), isinstance(rectBBox.height, float)))

            assert isinstance(detection.boundingBox.score, float), "score is not float"
            assert 0 <= detection.boundingBox.score < 1, "score out of range [0,1]"

    def test_bounding_box_as_dict(self):
        for detection in self.getDetections():
            assert jsonschema.validate(detection.boundingBox.asDict(), REQUIRED_FACE_DETECTION) is None, detection.asDict()
            assert detection.boundingBox.rect.isValid()

    def test_face_detection_as_dict(self):
        for detection in self.getDetections(detect5Landmarks=False, detect68Landmarks=False):
            assert jsonschema.validate(detection.asDict(), REQUIRED_FACE_DETECTION) is None, detection.asDict()
        for detection in self.getDetections(detect5Landmarks=True, detect68Landmarks=True):
            currentSchema = json.loads(json.dumps(detection.asDict()))
            assert jsonschema.validate(currentSchema, REQUIRED_FACE_DETECTION) is None, detection.asDict()

    def test_different_detectors(self):
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                for detectionFunction in ("detect", "detectOne"):
                    if detectionFunction == "detectOne":
                        detection = detector.detectOne(image=VLIMAGE_ONE_FACE)
                    else:
                        detection = detector.detect(images=[VLIMAGE_ONE_FACE])[0][0]
                    self.assertDetectionTrue(detection, [VLIMAGE_ONE_FACE])
                    assert score == pytest.approx(detection.boundingBox.score, rel=1e-4)

    def test_detect_one_faces_on_image_with_several_faces(self):
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=VLIMAGE_SEVERAL_FACE)
                self.assertDetectionTrue(detection, [VLIMAGE_SEVERAL_FACE])

    def test_detect_one_faces_on_image_without_faces(self):
        imageWithoutFace = VLImage.load(filename=NO_FACES)
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=imageWithoutFace)
                assert detection is None, detection

    def test_detect_image_without_faces(self):
        imageWithoutFace = VLImage.load(filename=NO_FACES)
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                detection = detector.detect(images=[imageWithoutFace])
                assert 0 == len(detection[0])

    def test_detect_by_area_one_face(self):
        area = Rect(0, 0, 100, 100)
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=area)
                assert detection is None, detection

                detection = detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=GOOD_AREA)
                self.assertDetectionTrue(detection, [VLIMAGE_ONE_FACE])

    def test_detect_several_faces(self):
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                detection = detector.detect(images=[VLIMAGE_SEVERAL_FACE])
                self.assertDetectionTrue(detection, [VLIMAGE_SEVERAL_FACE])
                assert 1 == len(detection)
                assert 5 == len(detection[0])

    def test_batch_detect(self):
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                detection = detector.detect(images=[VLIMAGE_SEVERAL_FACE, VLIMAGE_ONE_FACE])
                self.assertDetectionTrue(detection, [VLIMAGE_SEVERAL_FACE, VLIMAGE_ONE_FACE])
                assert 2 == len(detection)
                assert 5 == len(detection[0])
                assert 1 == len(detection[1])

    def test_detect_landmarks(self):
        Case = namedtuple("Case", ("detect5Landmarks", "detect68Landmarks"))
        cases = [
            Case(landmarks5, landmarks68) for landmarks5, landmarks68 in itertools.product((True, False), (True, False))
        ]
        for case in cases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                for detectionFunction in ("detect", "detectOne"):
                    with self.subTest(funcName=detectionFunction):
                        if detectionFunction == "detectOne":
                            detection = TestDetector.detector.detectOne(
                                image=VLIMAGE_ONE_FACE,
                                detect68Landmarks=case.detect68Landmarks,
                                detect5Landmarks=case.detect5Landmarks,
                            )
                        else:
                            detection = TestDetector.detector.detect(
                                images=[VLIMAGE_ONE_FACE],
                                detect68Landmarks=case.detect68Landmarks,
                                detect5Landmarks=case.detect5Landmarks,
                            )[0][0]

                        if case.detect5Landmarks:
                            assert isinstance(detection.landmarks5, Landmarks5), detection.landmarks5
                        else:
                            assert detection.landmarks5 is None, detection.landmarks5
                        if case.detect68Landmarks:
                            assert isinstance(detection.landmarks68, Landmarks68), detection.landmarks68
                        else:
                            assert detection.landmarks68 is None, detection.landmarks68

    def test_detect_limit(self):
        imageWithManyFaces = VLImage.load(filename=MANY_FACES)
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                detection = detector.detect(images=[imageWithManyFaces])[0]
                assert 5 == len(detection)

                detection = detector.detect(images=[imageWithManyFaces], limit=20)[0]
                if detector.detectorType.name == 'FACE_DET_V3':
                    assert 20 == len(detection)
                else:
                    assert 19 == len(detection)

    @pytest.mark.skip("core bug")
    def test_detect_limit_bad_param(self):
        imageWithManyFaces = VLImage.load(filename=MANY_FACES)
        detections = TestDetector.detector.detect(images=[imageWithManyFaces], limit=-1)[0]

    def test_detect_bad_image_color_format(self):
        imageWithOneFaces = VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.B8G8R8)
        errorDetail = "Bad image format for detection, format: B8G8R8, image: one_face.jpg"
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                for detectionFunction in ("detect", "detectOne"):
                    with pytest.raises(LunaSDKException) as exceptionInfo:
                        if detectionFunction == "detectOne":
                            detector.detectOne(image=imageWithOneFaces)
                        else:
                            detector.detect(images=[ImageForDetection(imageWithOneFaces, imageWithOneFaces.rect)])
                    self.assertLunaVlError(exceptionInfo, 100011,
                                           LunaVLError.InvalidImageFormat.format(details=errorDetail))

    def test_detect_by_area_and_not(self):
        areaWithoutFace = Rect(0, 0, 100, 100)
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                detection = TestDetector.detector.detect(images=[ImageForDetection(VLIMAGE_ONE_FACE, areaWithoutFace),
                                                                 ImageForDetection(VLIMAGE_ONE_FACE, GOOD_AREA),
                                                                 VLIMAGE_ONE_FACE])
                assert 3 == len(detection)
                assert 0 == len(detection[0])
                assert 1 == len(detection[1])
                assert 1 == len(detection[2])

    def test_detect_rect_is_out_of_bounds(self):
        badArea = Rect(100, 100, VLIMAGE_ONE_FACE.rect.width, VLIMAGE_ONE_FACE.rect.height)
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                for detectionFunction in ("detect", "detectOne"):
                    if detectionFunction == "detectOne":
                        with pytest.raises(LunaSDKException) as exceptionInfo:
                            detector.detectOne(VLIMAGE_ONE_FACE, detectArea=badArea)
                        self.assertLunaVlError(exceptionInfo, 100005, LunaVLError.Internal)
                    else:
                        with pytest.raises(LunaSDKException) as exceptionInfo:
                            detector.detect(images=[ImageForDetection(VLIMAGE_ONE_FACE, badArea)])
                        self.assertLunaVlError(exceptionInfo, 100016, LunaVLError.InvalidRect)

    def test_detect_excess_memory_usage(self):
        for typeImageForDetection in ("vlImage", "ImageForDetection"):
            with pytest.raises(LunaSDKException) as exceptionInfo:
                if typeImageForDetection == "vlImage":
                    TestDetector.detector.detect(images=[VLIMAGE_ONE_FACE] * 20)
                else:
                    TestDetector.detector.detect(images=[ImageForDetection(VLIMAGE_ONE_FACE, GOOD_AREA)] * 20)
            self.assertLunaVlError(exceptionInfo, 100005, LunaVLError.Internal)

    def test_detect_one_invalid_rectangle(self):
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detectOne(VLIMAGE_ONE_FACE, detectArea=Rect())
                self.assertLunaVlError(exceptionInfo, 100005, LunaVLError.Internal)

    def test_detect_invalid_rectangle(self):
        for subTest, detector, score in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detect(images=[ImageForDetection(VLIMAGE_ONE_FACE, Rect())])
                self.assertLunaVlError(exceptionInfo, 100016, LunaVLError.InvalidRect)
