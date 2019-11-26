import itertools
import json
from collections import namedtuple
from typing import Optional, Union, List, Tuple

import jsonschema
import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.faceengine.facedetector import (
    FaceDetector,
    Landmarks5,
    Landmarks68,
    FaceDetection,
    ImageForDetection)
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage, ColorFormat
from tests.base import BaseTestClass
from tests.resources import ONE_FACE, SEVERAL_FACES, MANY_FACES, NO_FACES
from tests.schemas import REQUIRED_FACE_DETECTION, LANDMARKS5, LANDMARKS68


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
    def getDetectExceptionInfo(imageVl: Union[List[VLImage], VLImage], detectArea: Optional[Rect] = None):
        with pytest.raises(LunaSDKException) as ex:
            if isinstance(imageVl, list):
                TestDetector.detector.detect(images=[ImageForDetection(imageVl[0], detectArea)])
            else:
                TestDetector.detector.detectOne(image=imageVl, detectArea=detectArea)
        return ex

    @staticmethod
    def getDetections(detect5Landmarks: bool = True,
                      detect68Landmarks: bool = False) -> Tuple[FaceDetection, FaceDetection]:
        oneFace = VLImage.load(filename=ONE_FACE)
        detectOne = TestDetector.detector.detectOne(image=oneFace, detect5Landmarks=detect5Landmarks,
                                                    detect68Landmarks=detect68Landmarks)
        detect = TestDetector.detector.detect(images=[oneFace], detect5Landmarks=detect5Landmarks,
                                              detect68Landmarks=detect68Landmarks)[0][0]
        return detectOne, detect

    def test_image_for_detection(self):
        oneFace = VLImage.load(filename=ONE_FACE)
        area = Rect(100, 100, oneFace.rect.width - 100, oneFace.rect.height - 100)
        detections = TestDetector.detector.detect(images=[ImageForDetection(oneFace, area)])
        assert 1 == len(detections)
        assert isinstance(detections[0][0], FaceDetection), detections[0][0]
        currentBBox = detections[0][0].boundingBox.asDict()
        assert jsonschema.validate(currentBBox, REQUIRED_FACE_DETECTION) is None, currentBBox

    def test_landmarks_class(self):
        for detection in self.getDetections(detect68Landmarks=True):
            assert isinstance(detection.landmarks5.points and detection.landmarks68.points, tuple), detection.asDict()
            assert 5 == len(detection.landmarks5.points)
            assert 68 == len(detection.landmarks68.points)

            currentLandmarks5 = json.loads(json.dumps(detection.landmarks5.asDict()))
            assert jsonschema.validate(currentLandmarks5, LANDMARKS5) is None, currentLandmarks5
            currentLandmarks68 = json.loads(json.dumps(detection.landmarks68.asDict()))
            assert jsonschema.validate(currentLandmarks68, LANDMARKS68) is None, currentLandmarks68

    def test_validate_bounding_box(self):
        for detect in self.getDetections():
            assert jsonschema.validate(detect.boundingBox.asDict(), REQUIRED_FACE_DETECTION) is None, detect.asDict()
            assert detect.boundingBox.rect.isValid()

    def test_face_detection_dict(self):
        for detect in self.getDetections(detect5Landmarks=False, detect68Landmarks=False):
            assert detect.landmarks5 is None, detect.landmarks5
            assert detect.landmarks68 is None, detect.landmarks68
            assert jsonschema.validate(detect.asDict(), REQUIRED_FACE_DETECTION) is None, detect.asDict()
        for detect in self.getDetections(detect5Landmarks=True, detect68Landmarks=True):
            currentSchema = json.loads(json.dumps(detect.asDict()))
            assert jsonschema.validate(currentSchema, REQUIRED_FACE_DETECTION) is None, detect.asDict()
            assert isinstance(detect.landmarks5, Landmarks5), detect.landmarks5
            assert isinstance(detect.landmarks68, Landmarks68), detect.landmarks68

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
                assert detection.image == image, detection.image
                assert detection.boundingBox.rect.isValid()
                assert case.score == pytest.approx(detection.boundingBox.score, rel=1e-4)

    def test_detect_one_face(self):
        oneFace = VLImage.load(filename=ONE_FACE)
        detection = TestDetector.detector.detectOne(image=oneFace)
        assert isinstance(detection, FaceDetection), detection
        assert detection.image == oneFace
        assert detection.boundingBox.rect.isValid()

    def test_detect_one_faces_on_image_with_several_faces(self):
        severalFace = VLImage.load(filename=SEVERAL_FACES)
        detection = TestDetector.detector.detectOne(image=severalFace)
        assert isinstance(detection, FaceDetection), detection

    def test_detect_one_faces_on_image_without_faces(self):
        imageWithoutFace = VLImage.load(filename=NO_FACES)
        detection = TestDetector.detector.detectOne(image=imageWithoutFace)
        assert detection is None, detection

    def test_detect_by_area_one_face(self):
        oneFace = VLImage.load(filename=ONE_FACE)
        area = Rect(0, 0, 100, 100)
        detection = TestDetector.detector.detectOne(image=oneFace, detectArea=area)
        assert detection is None, detection
        area = Rect(100, 100, oneFace.rect.width - 100, oneFace.rect.height - 100)
        detection = TestDetector.detector.detectOne(image=oneFace, detectArea=area)
        assert isinstance(detection, FaceDetection), detection

    def test_detect_several_faces(self):
        oneFace = VLImage.load(filename=SEVERAL_FACES)
        detections = TestDetector.detector.detect(images=[oneFace])
        assert 1 == len(detections)
        assert 5 == len(detections[0])

    def test_batch_detect(self):
        severalFace = VLImage.load(filename=SEVERAL_FACES)
        oneFace = VLImage.load(filename=ONE_FACE)
        detections = TestDetector.detector.detect(images=[severalFace, oneFace])
        assert 2 == len(detections)
        assert 5 == len(detections[0])
        assert 1 == len(detections[1])

    def test_detect_landmarks(self):
        oneFace = VLImage.load(filename=ONE_FACE)

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
                                image=oneFace,
                                detect68Landmarks=case.detect68Landmarks,
                                detect5Landmarks=case.detect5Landmarks,
                            )
                        else:
                            detection = TestDetector.detector.detect(
                                images=[oneFace],
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

    def test_detect_default_landmarks_params(self):
        image = VLImage.load(filename=ONE_FACE)
        for func in ("detect", "detectOne"):
            with self.subTest(funcName=func):
                if func == "detectOne":
                    detection = TestDetector.detector.detectOne(image=image)
                else:
                    detection = TestDetector.detector.detect(images=[image])[0][0]

                assert isinstance(detection.landmarks5, Landmarks5), detection.landmarks5
                assert detection.landmarks68 is None, detection.landmarks68

    def test_detect_limit(self):
        manyFaces = VLImage.load(filename=MANY_FACES)
        detections = TestDetector.detector.detect(images=[manyFaces])[0]
        assert 5 == len(detections)

        detections = TestDetector.detector.detect(images=[manyFaces], limit=20)[0]
        assert 19 == len(detections)

    @pytest.mark.skip("core bug")
    def test_detect_limit_bad_param(self):
        image = VLImage.load(filename=MANY_FACES)
        detections = TestDetector.detector.detect(images=[image], limit=-1)[0]

    def test_detect_bad_image_color_format(self):
        oneFace = VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.B8G8R8)
        errorDetail = "Bad image format for detection, format: B8G8R8, image: one_face.jpg"
        for func in ("detect", "detectOne"):
            with self.subTest(funcName=func):
                if func == "detectOne":
                    exceptionInfo = self.getDetectExceptionInfo(oneFace)
                else:
                    exceptionInfo = self.getDetectExceptionInfo([oneFace], detectArea=oneFace.rect)
                self.assertLunaVlError(exceptionInfo, 100011,
                                       LunaVLError.InvalidImageFormat.format(details=errorDetail))

    def test_detect_by_area_and_not(self):
        oneFace = VLImage.load(filename=ONE_FACE)
        area1 = Rect(0, 0, 100, 100)
        area2 = Rect(100, 100, oneFace.rect.width - 100, oneFace.rect.height - 100)
        detections = TestDetector.detector.detect(
            images=[ImageForDetection(oneFace, area1), ImageForDetection(oneFace, area2), oneFace],
        )
        assert 3 == len(detections)
        assert 0 == len(detections[0])
        assert 1 == len(detections[1])
        assert 1 == len(detections[2])

    def test_detect_by_bad_area(self):
        oneFace = VLImage.load(filename=ONE_FACE)
        area = Rect(100, 100, oneFace.rect.width, oneFace.rect.height)
        exceptionInfo = self.getDetectExceptionInfo(oneFace, detectArea=area)
        self.assertLunaVlError(exceptionInfo, 100005, LunaVLError.Internal)

    def test_detect_excess_memory_usage(self):
        oneFace = VLImage.load(filename=ONE_FACE)
        for item in ("default", "ImageForDetection"):
            with pytest.raises(LunaSDKException) as exceptionInfo:
                if item == "default":
                    TestDetector.detector.detect(images=[oneFace] * 20)
                else:
                    TestDetector.detector.detect(images=[ImageForDetection(oneFace, oneFace.rect)] * 20)
            self.assertLunaVlError(exceptionInfo, 100005, LunaVLError.Internal)

    def test_detect_one_invalid_rectangle(self):
        oneFace = VLImage.load(filename=ONE_FACE)
        with pytest.raises(LunaSDKException) as exceptionInfo:
            TestDetector.detector.detectOne(image=oneFace, detectArea=Rect())
        self.assertLunaVlError(exceptionInfo, 100005, LunaVLError.Internal)

    def test_detect_invalid_rectangle(self):
        oneFace = VLImage.load(filename=ONE_FACE)
        with pytest.raises(LunaSDKException) as exceptionInfo:
            TestDetector.detector.detect(images=[ImageForDetection(oneFace, Rect())])
        self.assertLunaVlError(exceptionInfo, 100016, LunaVLError.InvalidRect)
