import itertools
from collections import namedtuple
from typing import Optional

import pytest

from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.faceengine.facedetector import (
    FaceDetector,
    Landmarks5,
    Landmarks68,
    FaceDetection,
    ImageForRedetection)
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ONE_FACE, SEVERAL_FACES


class TestDetector(BaseTestClass):
    """
    Test of redetection.
    """

    detector: FaceDetector = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)

    @staticmethod
    def getErrorRedetectOne(imageVl: VLImage, detection: Optional[FaceDetection] = None, bBox: Optional[Rect] = None):
        with pytest.raises(LunaSDKException) as ex:
            if bBox is None:
                TestDetector.detector.redetectOne(image=imageVl, detection=detection)
            else:
                TestDetector.detector.redetectOne(image=imageVl, bBox=bBox)
        error = {"error_code": ex.value.error.errorCode, "desc": ex.value.error.description,
                 "detail": ex.value.error.detail}
        return error

    @staticmethod
    def getErrorRedetect(imageVl: VLImage, bBoxes):
        with pytest.raises(LunaSDKException) as ex:
            TestDetector.detector.redetect(images=[ImageForRedetection(image=imageVl, bBoxes=bBoxes)])
        error = {"error_code": ex.value.error.errorCode, "desc": ex.value.error.description,
                 "detail": ex.value.error.detail}
        return error

    def test_detect_landmarks(self):
        image = VLImage.load(filename=ONE_FACE)
        detectOne = TestDetector.detector.detectOne(image=image)
        Case = namedtuple("Case", ("detect5Landmarks", "detect68Landmarks"))
        cases = [
            Case(landmarks5, landmarks68) for landmarks5, landmarks68 in itertools.product((True, False), (True, False))
        ]

        for case in cases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                for func in ("redetect", "redetectOne"):
                    with self.subTest(funcName=func):
                        if func == "redetectOne":
                            response = TestDetector.detector.redetectOne(
                                image=image,
                                detection=detectOne,
                                detect68Landmarks=case.detect68Landmarks,
                                detect5Landmarks=case.detect5Landmarks,
                            )
                        else:
                            response = TestDetector.detector.redetect(
                                images=[ImageForRedetection(image=image, bBoxes=[detectOne.boundingBox.rect])],
                                detect68Landmarks=case.detect68Landmarks,
                                detect5Landmarks=case.detect5Landmarks,
                            )[0][0]

                        if case.detect5Landmarks:
                            assert isinstance(response.landmarks5, Landmarks5)
                        else:
                            assert response.landmarks5 is None
                        if case.detect68Landmarks:
                            assert isinstance(response.landmarks68, Landmarks68)
                        else:
                            assert response.landmarks68 is None

    def test_redetect_one_image(self):
        image = VLImage.load(filename=ONE_FACE)
        detection = TestDetector.detector.detectOne(image=image)
        for parameter in ("bBox", "detection"):
            if parameter == "bBox":
                response = TestDetector.detector.redetectOne(image=image,
                                                             bBox=detection.boundingBox.rect)
            else:
                response = TestDetector.detector.redetectOne(image=image,
                                                             detection=detection)
        assert isinstance(response, FaceDetection)

    def test_batch_redetect(self):
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        imageWithSeveralFace = VLImage.load(filename=SEVERAL_FACES)
        detectOne = TestDetector.detector.detectOne(image=imageWithOneFace)
        detectSeveral = TestDetector.detector.detect(images=[imageWithSeveralFace])
        redetect = TestDetector.detector.redetect(images=[ImageForRedetection(image=imageWithSeveralFace,
                                                                              bBoxes=[face.boundingBox.rect
                                                                                      for face in detectSeveral[0]]),
                                                          ImageForRedetection(image=imageWithOneFace,
                                                                              bBoxes=[detectOne.boundingBox.rect])])
        [isinstance(face, FaceDetection) for face in (*redetect[0], *redetect[1])]
        assert 2 == len(redetect)
        assert 5 == len(redetect[0])
        assert 1 == len(redetect[1])

    def test_redetect_face_with_wrong_area(self):
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        redetectOne = TestDetector.detector.redetectOne(image=imageWithOneFace, bBox=Rect(0, 0, 100, 100))
        redetect = TestDetector.detector.redetect(images=[ImageForRedetection(image=imageWithOneFace,
                                                                              bBoxes=[Rect(0, 0, 100, 100)])])
        assert redetectOne is None
        assert redetect[0][0] is None

    def test_redetect_face_with_invalid_rect(self):
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        error = self.getErrorRedetectOne(imageVl=imageWithOneFace, bBox=Rect())
        assert error["error_code"] == 100016
        assert error["desc"] == "Invalid rectangle"
        assert error["detail"] == "Invalid rectangle"

    def test_redetect_face_without_detection_and_bbox(self):
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        error = self.getErrorRedetectOne(imageVl=imageWithOneFace)
        assert error["error_code"] == 110019
        assert error["desc"] == "Detect one face error"
        assert error["detail"] == ""

    def test_redetect_unknown_fsdk_error(self):
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        error = self.getErrorRedetect(imageVl=imageWithOneFace, bBoxes=[Rect()])
        assert error["error_code"] == 99999
        assert error["desc"] == "Unknown fsdk core error"
        assert error["detail"] == "Unknown error"
