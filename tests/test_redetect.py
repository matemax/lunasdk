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
INVALID_RECT = Rect(0, 0, 0, 0)
ERROR_CORE_RECT = Rect(0.1, 0.1, 0.1, 0.1)  # anything out of range (0.1, 1)


class TestDetector(BaseTestClass):
    """
    Test of redetection.
    """

    detector: FaceDetector = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)

    def test_get_landmarks_for_redetect_one(self):
        """
        Test get and check landmark instances for re-detection of one face
        """
        detectOne = TestDetector.detector.detectOne(image=VLIMAGE_ONE_FACE)
        for case in self.landmarksCases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                redetect = TestDetector.detector.redetectOne(image=VLIMAGE_ONE_FACE, detection=detectOne,
                                                             detect68Landmarks=case.detect68Landmarks,
                                                             detect5Landmarks=case.detect5Landmarks)
                if case.detect5Landmarks:
                    assert isinstance(redetect.landmarks5, Landmarks5)
                else:
                    assert redetect.landmarks5 is None
                if case.detect68Landmarks:
                    assert isinstance(redetect.landmarks68, Landmarks68)
                else:
                    assert redetect.landmarks68 is None

    def test_get_landmarks_for_batch_redetect(self):
        """
        Test get and check landmark instances for batch re-detection
        """
        detectOne = TestDetector.detector.detectOne(image=VLIMAGE_ONE_FACE)
        for case in self.landmarksCases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                redetect = TestDetector.detector.redetect(images=[
                    ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[detectOne.boundingBox.rect])],
                    detect68Landmarks=case.detect68Landmarks,
                    detect5Landmarks=case.detect5Landmarks)[0][0]
                if case.detect5Landmarks:
                    assert isinstance(redetect.landmarks5, Landmarks5)
                else:
                    assert redetect.landmarks5 is None
                if case.detect68Landmarks:
                    assert isinstance(redetect.landmarks68, Landmarks68)
                else:
                    assert redetect.landmarks68 is None

    def test_redetect_one_with_bbox_option(self):
        """
        Test re-detection of one face with bounding box option
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE)
                redetect = detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=detection.boundingBox.rect)
                self.assertFaceDetection(redetect, VLIMAGE_ONE_FACE)

    def test_redetect_one_with_detection_option(self):
        """
        Test re-detection of one face with detection options
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE)
                redetect = detector.redetectOne(image=VLIMAGE_ONE_FACE, detection=detection)
                self.assertFaceDetection(redetect, VLIMAGE_ONE_FACE)

    def test_redetect_with_one_face(self):
        """
        Test re-detection with one face image
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE)
                redetect = detector.redetect(images=[ImageForRedetection(image=VLIMAGE_ONE_FACE,
                                                                         bBoxes=[detection.boundingBox.rect])])[0]
                self.assertFaceDetection(redetect, VLIMAGE_ONE_FACE)

    def test_batch_redetect(self):
        """
        Test re-detection batch of images
        """
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

    def test_redetect_by_area_without_face(self):
        """
        Test re-detection by area without face
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                redetectOne = detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=Rect(0, 0, 100, 100))
                redetect = detector.redetect(images=[ImageForRedetection(image=VLIMAGE_ONE_FACE,
                                                                         bBoxes=[Rect(0, 0, 100, 100)])])[0][0]
                assert redetectOne is None
                assert redetect is None

    def test_redetect_one_invalid_rectangle(self):
        """
        Test re-detection of one face with an invalid rect
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=INVALID_RECT)
                if detector.detectorType.name == 'FACE_DET_V3':
                    self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidRect)
                else:
                    self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidInput)

    def test_redetect_invalid_rectangle(self):
        """
        Test batch re-detection with an invalid rect
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.redetect(images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[INVALID_RECT])])
                self.assertLunaVlError(exceptionInfo, LunaVLError.UnknownError)

    def test_redetect_one_without_detection_and_bbox(self):
        """
        Test re-detection of one face without bounding box and face detection
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.redetectOne(image=VLIMAGE_ONE_FACE)
                self.assertLunaVlError(exceptionInfo, LunaVLError.DetectFacesError)

    @pytest.mark.skip("core bug")
    def test_rect_float(self):
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detector.redetect(images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[ERROR_CORE_RECT])])
