import itertools
import os
from collections import namedtuple
from typing import Optional

import pytest

from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.faceengine.facedetector import (
    FaceDetector,
    Landmarks5,
    Landmarks68,
    FaceDetection,
    ImageForDetection,
    BoundingBox)
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage, ColorFormat
from tests.base import BaseTestClass
from tests.resources import ONE_FACE, SEVERAL_FACES, MANY_FACES, NO_FACES

TEXT_FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "test_type.txt")


class TestDetector(BaseTestClass):
    """
    Test of detector.
    """

    detector: FaceDetector = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        try:
            os.remove(TEXT_FILE_PATH)
        except FileNotFoundError:
            pass

    @staticmethod
    def createTextFile():
        with open(TEXT_FILE_PATH, "w") as w:
            w.write("test type image")
        if os.path.isfile(TEXT_FILE_PATH):
            return TEXT_FILE_PATH

    @staticmethod
    def getErrorDetectOne(imageVl: VLImage, detectArea: Optional[Rect] = None):
        with pytest.raises(LunaSDKException) as ex:
            TestDetector.detector.detectOne(image=imageVl, detectArea=detectArea)
        error = {"error_code": ex.value.error.errorCode, "desc": ex.value.error.description,
                 "detail": ex.value.error.detail}
        return error

    @staticmethod
    def getErrorDetect(imageVl: VLImage, detectArea: Optional[Rect] = None):
        with pytest.raises(LunaSDKException) as ex:
            TestDetector.detector.detect(images=[ImageForDetection(imageVl, detectArea)])
        error = {"error_code": ex.value.error.errorCode, "desc": ex.value.error.description,
                 "detail": ex.value.error.detail}
        return error

    def test_detect_one_face(self):
        image = VLImage.load(filename=ONE_FACE)
        detection = TestDetector.detector.detectOne(image=image)
        assert detection.image == image
        assert detection.boundingBox.rect.isValid()

    def test_different_detectors(self):
        image = VLImage.load(filename=ONE_FACE)

        Case = namedtuple("Case", ("type", "score"))

        cases = [
            Case(DetectorType.FACE_DET_V1, 0.9983),
            Case(DetectorType.FACE_DET_V2, 0.9913),
            Case(DetectorType.FACE_DET_V3, 0.9999),
        ]

        for case in cases:
            with self.subTest(detectorType=case.type):
                detector = TestDetector.faceEngine.createFaceDetector(case.type)

                detection = detector.detectOne(image=image)
                assert detection.image == image
                assert detection.boundingBox.rect.isValid()
                assert case.score == pytest.approx(detection.boundingBox.score, rel=1e-4)

    def test_detect_several_faces(self):
        image = VLImage.load(filename=SEVERAL_FACES)
        detections = TestDetector.detector.detect(images=[image])
        assert 1 == len(detections)
        assert 5 == len(detections[0])

    def test_batch_detect(self):
        image1 = VLImage.load(filename=SEVERAL_FACES)
        image2 = VLImage.load(filename=ONE_FACE)
        detections = TestDetector.detector.detect(images=[image1, image2])
        assert 2 == len(detections)
        assert 5 == len(detections[0])
        assert 1 == len(detections[1])

    def test_detect_landmarks(self):
        image = VLImage.load(filename=ONE_FACE)

        Case = namedtuple("Case", ("detect5Landmarks", "detect68Landmarks"))
        cases = [
            Case(landmarks5, landmarks68) for landmarks5, landmarks68 in itertools.product((True, False), (True, False))
        ]

        for case in cases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                for func in ("detect", "detectOne"):
                    with self.subTest(funcName=func):
                        if func == "detectOne":
                            detection = TestDetector.detector.detectOne(
                                image=image,
                                detect68Landmarks=case.detect68Landmarks,
                                detect5Landmarks=case.detect5Landmarks,
                            )

                        else:
                            detection = TestDetector.detector.detect(
                                images=[image],
                                detect68Landmarks=case.detect68Landmarks,
                                detect5Landmarks=case.detect5Landmarks,
                            )[0][0]

                        if case.detect5Landmarks:
                            assert isinstance(detection.landmarks5, Landmarks5)
                        else:
                            assert detection.landmarks5 is None
                        if case.detect68Landmarks:
                            assert isinstance(detection.landmarks68, Landmarks68)
                        else:
                            assert detection.landmarks68 is None

    def test_detect_default_landmarks_params(self):
        image = VLImage.load(filename=ONE_FACE)

        for func in ("detect", "detectOne"):
            with self.subTest(funcName=func):
                if func == "detectOne":
                    detection = TestDetector.detector.detectOne(image=image)

                else:
                    detection = TestDetector.detector.detect(images=[image])[0][0]

                assert isinstance(detection.landmarks5, Landmarks5)
                assert detection.landmarks68 is None

    def test_detect_limit(self):
        image = VLImage.load(filename=MANY_FACES)
        detections = TestDetector.detector.detect(images=[image])[0]
        assert 5 == len(detections)

        detections = TestDetector.detector.detect(images=[image], limit=20)[0]
        assert 20 == len(detections)

    @pytest.mark.skip("core bug")
    def test_detect_limit_bad_param(self):
        image = VLImage.load(filename=MANY_FACES)
        detections = TestDetector.detector.detect(images=[image], limit=-1)[0]

    def test_detect_one_faces_on_image_with_several_faces(self):
        image = VLImage.load(filename=SEVERAL_FACES)
        detection = TestDetector.detector.detectOne(image=image)
        assert isinstance(detection, FaceDetection)

    def test_detect_one_faces_on_image_without_faces(self):
        image = VLImage.load(filename=NO_FACES)
        detection = TestDetector.detector.detectOne(image=image)
        assert detection is None

    def test_detect_bad_image_type(self):
        with pytest.raises(LunaSDKException) as ex:
            VLImage.load(filename=self.createTextFile())
        assert ex.value.error.errorCode == 100033
        assert ex.value.error.description == "Unsupported image type"
        assert ex.value.error.detail == "Unsupported type"

    def test_detect_bad_image_color_format(self):
        image = VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.B8G8R8)

        for func in ("detect", "detectOne"):
            with self.subTest(funcName=func):
                if func == "detectOne":
                    error = self.getErrorDetectOne(image)
                else:
                    error = self.getErrorDetect(image, detectArea=image.rect)
                assert error["error_code"] == 100011
                assert error["desc"] == "Invalid image format"
                assert error["detail"] == "Bad image format for detection, format: B8G8R8, image: one_face.jpg"

    def test_detect_by_area_one_face(self):
        image = VLImage.load(filename=ONE_FACE)
        area = Rect(0, 0, 100, 100)
        detection = TestDetector.detector.detectOne(image=image, detectArea=area)
        assert detection is None
        area = Rect(100, 100, image.rect.width - 100, image.rect.height - 100)
        detection = TestDetector.detector.detectOne(image=image, detectArea=area)
        assert isinstance(detection, FaceDetection)

    def test_detect_by_area_and_not(self):
        image = VLImage.load(filename=ONE_FACE)
        area1 = Rect(0, 0, 100, 100)
        area2 = Rect(100, 100, image.rect.width - 100, image.rect.height - 100)
        detections = TestDetector.detector.detect(
            images=[ImageForDetection(image, area1), ImageForDetection(image, area2), image]
        )
        assert 3 == len(detections)
        assert 0 == len(detections[0])
        assert 1 == len(detections[1])
        assert 1 == len(detections[2])

    def test_detect_by_bad_area(self):
        image = VLImage.load(filename=ONE_FACE)
        area = Rect(100, 100, image.rect.width, image.rect.height)
        error = self.getErrorDetectOne(image, detectArea=area)
        assert error["error_code"] == 100005
        assert error["desc"] == "Internal error"
        assert error["detail"] == "Internal error"

    def test_check_bounding_box(self):
        image = VLImage.load(filename=ONE_FACE)
        detection = TestDetector.detector.detectOne(image=image).boundingBox
        assert isinstance(detection, BoundingBox)
        assert detection.rect.isValid()
