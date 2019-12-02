import itertools
from collections import namedtuple

import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.faceengine.facedetector import (
    FaceDetector,
    Landmarks5,
    Landmarks68,
    ImageForRedetection)
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import SEVERAL_FACES, CLEAN_ONE_FACE

VLIMAGE_ONE_FACE = VLImage.load(filename=CLEAN_ONE_FACE)
VLIMAGE_SEVERAL_FACE = VLImage.load(filename=SEVERAL_FACES)


class TestDetector(BaseTestClass):
    """
    Test of redetection.
    """

    detector: FaceDetector = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)

    def test_redetect_landmarks_option(self):
        detectOne = TestDetector.detector.detectOne(image=VLIMAGE_ONE_FACE)

        Case = namedtuple("Case", ("detect5Landmarks", "detect68Landmarks"))
        cases = [
            Case(landmarks5, landmarks68) for landmarks5, landmarks68 in itertools.product((True, False), (True, False))
        ]
        for case in cases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                for redetectFunction in ("redetect", "redetectOne"):
                    with self.subTest(funcName=redetectFunction):
                        if redetectFunction == "redetectOne":
                            response = TestDetector.detector.redetectOne(
                                image=VLIMAGE_ONE_FACE,
                                detection=detectOne,
                                detect68Landmarks=case.detect68Landmarks,
                                detect5Landmarks=case.detect5Landmarks,
                            )
                        else:
                            response = TestDetector.detector.redetect(
                                images=[ImageForRedetection(image=VLIMAGE_ONE_FACE,
                                                            bBoxes=[detectOne.boundingBox.rect])],
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

    def test_redetect_one_with_different_options(self):
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE)
                for optionDetect in ("bBox", "detection"):
                    if optionDetect == "bBox":
                        redetect = detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=detection.boundingBox.rect)
                    else:
                        redetect = detector.redetectOne(image=VLIMAGE_ONE_FACE, detection=detection)
                    self.assertFaceDetection(redetect, VLIMAGE_ONE_FACE)

    def test_redetect_with_one_face(self):
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE)
                redetect = detector.redetect(images=[ImageForRedetection(image=VLIMAGE_ONE_FACE,
                                                                         bBoxes=[detection.boundingBox.rect])])[0]
                self.assertFaceDetection(redetect, VLIMAGE_ONE_FACE)

    def test_batch_redetect(self):
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detectOne = detector.detectOne(image=VLIMAGE_ONE_FACE)
                detectSeveral = detector.detect(images=[VLIMAGE_SEVERAL_FACE])
                redetect = detector.redetect(images=[ImageForRedetection(image=VLIMAGE_SEVERAL_FACE,
                                                                         bBoxes=[face.boundingBox.rect
                                                                                 for face in detectSeveral[0]]),
                                                     ImageForRedetection(image=VLIMAGE_ONE_FACE,
                                                                         bBoxes=[detectOne.boundingBox.rect])])
                self.assertFaceDetection(redetect[0], VLIMAGE_SEVERAL_FACE)
                self.assertFaceDetection(redetect[1], VLIMAGE_ONE_FACE)
                assert 2 == len(redetect)
                assert 5 == len(redetect[0])
                assert 1 == len(redetect[1])

    def test_redetect_face_with_wrong_area(self):
        for subTest, detector in self.detectorSubTest():
            with subTest:
                redetectOne = detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=Rect(0, 0, 100, 100))
                redetect = detector.redetect(images=[ImageForRedetection(image=VLIMAGE_ONE_FACE,
                                                                         bBoxes=[Rect(0, 0, 100, 100)])])[0][0]
                assert redetectOne is None
                assert redetect is None

    def test_redetect_one_face_invalid_rectangle(self):
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=Rect(0.666, 0.666, 0.666, 0.666))
                if detector.detectorType.name == 'FACE_DET_V3':
                    self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidRect)
                else:
                    self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidInput)

    def test_redetect_face_invalid_rectangle(self):
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.redetect(images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[Rect()])])
                self.assertLunaVlError(exceptionInfo, LunaVLError.UnknownError)

    def test_redetect_face_without_detection_and_bbox(self):
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.redetectOne(image=VLIMAGE_ONE_FACE)
                self.assertLunaVlError(exceptionInfo, LunaVLError.DetectFacesError)
