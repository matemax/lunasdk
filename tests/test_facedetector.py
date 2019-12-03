import json

import jsonschema
import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.faceengine.facedetector import (
    FaceDetector,
    Landmarks5,
    Landmarks68,
    ImageForDetection)
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage, ColorFormat
from tests.base import BaseTestClass
from tests.resources import ONE_FACE, SEVERAL_FACES, MANY_FACES, NO_FACES
from tests.schemas import REQUIRED_FACE_DETECTION, LANDMARKS5

VLIMAGE_ONE_FACE = VLImage.load(filename=ONE_FACE)
VLIMAGE_SEVERAL_FACE = VLImage.load(filename=SEVERAL_FACES)
GOOD_AREA = Rect(100, 100, VLIMAGE_ONE_FACE.rect.width - 100, VLIMAGE_ONE_FACE.rect.height - 100)
OUTSIDE_AREA = Rect(100, 100, VLIMAGE_ONE_FACE.rect.width, VLIMAGE_ONE_FACE.rect.height)
AREA_WITHOUT_FACE = Rect(50, 50, 100, 100)


class TestDetector(BaseTestClass):
    """
    Test of detector.
    """

    detector: FaceDetector = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)

    def test_image_detection_with_transfer_option(self):
        """
        Test structure image for detection
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detect(images=[ImageForDetection(VLIMAGE_ONE_FACE, GOOD_AREA)])
                self.assertFaceDetection(detection[0], VLIMAGE_ONE_FACE)
                assert 1 == len(detection)

    def test_check_landmarks_points(self):
        """
        Test validation landmarks points
        """
        detection = TestDetector.detector.detectOne(image=VLIMAGE_ONE_FACE, detect68Landmarks=True)
        self.assertFaceDetection(detection, VLIMAGE_ONE_FACE)

        self.assertLandmarksPoints(detection.landmarks5.points)
        self.assertLandmarksPoints(detection.landmarks68.points)
        assert 5 == len(detection.landmarks5.points)
        assert 68 == len(detection.landmarks68.points)

    def test_landmarks_as_dict(self):
        """
        Test conversion landmarks to dictionary
        """
        detection = TestDetector.detector.detectOne(image=VLIMAGE_ONE_FACE)

        currentLandmarks5 = json.loads(json.dumps(detection.landmarks5.asDict()))
        assert jsonschema.validate(currentLandmarks5, LANDMARKS5) is None, \
            f"{currentLandmarks5} does not match with schema {LANDMARKS5}"

    def test_valid_bounding_box(self):
        """
        Test validate bounding box attributes (rect and score)
        """
        detection = TestDetector.detector.detectOne(image=VLIMAGE_ONE_FACE)
        self.assertFaceDetection(detection, VLIMAGE_ONE_FACE)

        self.checkRectAttr(detection.boundingBox.rect)
        assert isinstance(detection.boundingBox.score, float), "score is not float"
        assert 0 <= detection.boundingBox.score < 1, "score out of range [0,1]"

    def test_bounding_box_as_dict(self):
        """
        Test conversion bounding box to dictionary
        """
        boundingBox = TestDetector.detector.detectOne(image=VLIMAGE_ONE_FACE).boundingBox

        assert jsonschema.validate(boundingBox.asDict(), REQUIRED_FACE_DETECTION) is None, \
            f"{boundingBox.asDict()} does not match with schema {REQUIRED_FACE_DETECTION}"
        assert boundingBox.rect.isValid()

    def test_face_detection_as_dict(self):
        """
        Test conversion result face detection to dictionary
        """
        detectAsDict = TestDetector.detector.detectOne(image=VLIMAGE_ONE_FACE, detect5Landmarks=True,
                                                       detect68Landmarks=True).asDict()
        currentSchema = json.loads(json.dumps(detectAsDict))
        assert jsonschema.validate(currentSchema, REQUIRED_FACE_DETECTION) is None, \
            f"{currentSchema} does not match with schema {REQUIRED_FACE_DETECTION}"

    def test_detection_with_default_detector_type(self):
        """
        Test face detection with default detector type
        """
        for detectionFunction in ("detect", "detectOne"):
            with self.subTest(detectionFunction=detectionFunction):
                if detectionFunction == "detectOne":
                    detection = TestDetector.detector.detectOne(image=VLIMAGE_ONE_FACE)
                else:
                    detection = TestDetector.detector.detect(images=[VLIMAGE_ONE_FACE])[0]
                self.assertFaceDetection(detection, VLIMAGE_ONE_FACE)

    def test_detect_one_using_different_type_detector(self):
        """
        Test detection of one face using different type of detector
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE)
                self.assertFaceDetection(detection, VLIMAGE_ONE_FACE)

    def test_batch_detect_using_different_type_detector(self):
        """
        Test batch detection using different type of detector
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detect(images=[VLIMAGE_ONE_FACE])[0]
                self.assertFaceDetection(detection, VLIMAGE_ONE_FACE)

    def test_detect_one_with_image_of_several_faces(self):
        """
        Test detection of one face with image of several faces
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=VLIMAGE_SEVERAL_FACE)
                self.assertFaceDetection(detection, VLIMAGE_SEVERAL_FACE)

    def test_detect_one_with_image_without_faces(self):
        """
        Test detection of one face with image without faces
        """
        imageWithoutFace = VLImage.load(filename=NO_FACES)
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=imageWithoutFace)
                assert detection is None, detection

    def test_batch_detect_with_image_without_faces(self):
        """
        Test batch face detection with image without faces
        """
        imageWithoutFace = VLImage.load(filename=NO_FACES)
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detect(images=[imageWithoutFace])
                assert 0 == len(detection[0])

    def test_detect_one_by_area(self):
        """
        Test detection of one face by area with and without face
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=AREA_WITHOUT_FACE)
                assert detection is None, detection

                detection = detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=GOOD_AREA)
                self.assertFaceDetection(detection, VLIMAGE_ONE_FACE)

    def test_batch_detect_with_image_of_several_faces(self):
        """
        Test batch face detection with image of several faces
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detect(images=[VLIMAGE_SEVERAL_FACE])
                self.assertFaceDetection(detection[0], VLIMAGE_SEVERAL_FACE)
                assert 1 == len(detection)
                assert 5 == len(detection[0])

    def test_batch_detect_of_multiple_images(self):
        """
        Test batch detection of multiple images
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detect(images=[VLIMAGE_SEVERAL_FACE, VLIMAGE_ONE_FACE])
                self.assertFaceDetection(detection[0], VLIMAGE_SEVERAL_FACE)
                self.assertFaceDetection(detection[1], VLIMAGE_ONE_FACE)
                assert 2 == len(detection)
                assert 5 == len(detection[0])
                assert 1 == len(detection[1])

    def test_get_landmarks_for_detect_one(self):
        """
        Test get and check landmark instances for detection of one face
        """
        for case in self.landmarksCases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                detection = TestDetector.detector.detectOne(image=VLIMAGE_ONE_FACE,
                                                            detect68Landmarks=case.detect68Landmarks,
                                                            detect5Landmarks=case.detect5Landmarks)
                if case.detect5Landmarks:
                    assert isinstance(detection.landmarks5, Landmarks5), detection.landmarks5
                else:
                    assert detection.landmarks5 is None, detection.landmarks5
                if case.detect68Landmarks:
                    assert isinstance(detection.landmarks68, Landmarks68), detection.landmarks68
                else:
                    assert detection.landmarks68 is None, detection.landmarks68

    def test_get_landmarks_for_batch_detect(self):
        """
        Test get and check landmark instances for batch detect
        """
        for case in self.landmarksCases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                detection = TestDetector.detector.detect(images=[VLIMAGE_ONE_FACE],
                                                         detect68Landmarks=case.detect68Landmarks,
                                                         detect5Landmarks=case.detect5Landmarks)[0][0]
                if case.detect5Landmarks:
                    assert isinstance(detection.landmarks5, Landmarks5), detection.landmarks5
                else:
                    assert detection.landmarks5 is None, detection.landmarks5
                if case.detect68Landmarks:
                    assert isinstance(detection.landmarks68, Landmarks68), detection.landmarks68
                else:
                    assert detection.landmarks68 is None, detection.landmarks68

    def test_batch_detect_limit(self):
        """
        Test checking detection limit for an image
        """
        imageWithManyFaces = VLImage.load(filename=MANY_FACES)
        for subTest, detector in self.detectorSubTest():
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

    def test_detect_one_invalid_image_format(self):
        """
        Test invalid image format detection
        """
        imageWithOneFaces = VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.B8G8R8)
        errorDetail = "Bad image format for detection, format: B8G8R8, image: one_face.jpg"
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detectOne(image=imageWithOneFaces)
                self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidImageFormat.format(details=errorDetail))

    def test_batch_detect_invalid_image_format(self):
        """
        Test invalid image format detection
        """
        imageWithOneFaces = VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.B8G8R8)
        errorDetail = "Bad image format for detection, format: B8G8R8, image: one_face.jpg"
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detect(images=[ImageForDetection(imageWithOneFaces, GOOD_AREA)])
                self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidImageFormat.format(details=errorDetail))

    def test_batch_detect_by_area(self):
        """
        Test batch face detection by area with and without face
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                detection = detector.detect(images=[ImageForDetection(VLIMAGE_ONE_FACE, AREA_WITHOUT_FACE),
                                                    ImageForDetection(VLIMAGE_ONE_FACE, GOOD_AREA),
                                                    VLIMAGE_ONE_FACE])
                assert 3 == len(detection)
                assert 0 == len(detection[0])
                assert 1 == len(detection[1])
                assert 1 == len(detection[2])

    def test_bad_area_detection(self):
        """
        Test detection of one face in area outside image
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detectOne(VLIMAGE_ONE_FACE, detectArea=OUTSIDE_AREA)
                self.assertLunaVlError(exceptionInfo, LunaVLError.Internal)

    def test_batch_detect_in_area_outside_image(self):
        """
        Test batch detection in area outside image
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detect(images=[ImageForDetection(VLIMAGE_ONE_FACE, OUTSIDE_AREA)])
                self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidRect)

    def test_excessive_image_list_detection(self):
        """
        Test excessive image list detection
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            TestDetector.detector.detect(images=[VLIMAGE_ONE_FACE] * 20)
        self.assertLunaVlError(exceptionInfo, LunaVLError.Internal)

    def test_detect_one_invalid_rectangle(self):
        """
        Test detection of one face with an invalid rect
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detectOne(VLIMAGE_ONE_FACE, detectArea=Rect())
                self.assertLunaVlError(exceptionInfo, LunaVLError.Internal)

    def test_batch_detect_invalid_rectangle(self):
        """
        Test batch face detection with an invalid rect
        """
        for subTest, detector in self.detectorSubTest():
            with subTest:
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.detect(images=[ImageForDetection(VLIMAGE_ONE_FACE, Rect())])
                self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidRect)
