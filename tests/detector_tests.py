from collections import namedtuple

import pytest

from lunavl.sdk.faceengine.facedetector import DetectorType, FaceDetector
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass


class TestDetector(BaseTestClass):
    """
    Test of detector.
    """
    detector: FaceDetector = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)

    def test_detect_one_face(self):
        image = VLImage.load(filename='C:/temp/test.jpg')
        detection = TestDetector.detector.detectOne(image=image)
        assert detection.image == image
        assert detection.boundingBox.rect.isValid()

    def test_different_detectors(self):
        image = VLImage.load(filename='C:/temp/test.jpg')

        Case = namedtuple("Case", ("type", "score"))

        cases = [Case(DetectorType.FACE_DET_V1, 0.9983), Case(DetectorType.FACE_DET_V2, 0.9913),
                 Case(DetectorType.FACE_DET_V3, 0.9999)]

        for case in cases:
            with self.subTest(detectorType=case.type):
                detector = TestDetector.faceEngine.createFaceDetector(case.type)

                detection = detector.detectOne(image=image)
                assert detection.image == image
                assert detection.boundingBox.rect.isValid()
                assert case.score == pytest.approx(detection.boundingBox.score, rel=1e-4)
