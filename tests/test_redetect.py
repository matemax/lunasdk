import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.faceengine.facedetector import FaceDetector, ImageForRedetection
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from tests.detect_test_class import DetectTestClass
from tests.resources import SEVERAL_FACES, CLEAN_ONE_FACE, SMALL_IMAGE

VLIMAGE_SMALL = VLImage.load(filename=SMALL_IMAGE)
VLIMAGE_ONE_FACE = VLImage.load(filename=CLEAN_ONE_FACE)
VLIMAGE_SEVERAL_FACE = VLImage.load(filename=SEVERAL_FACES)
INVALID_RECT = Rect(0, 0, 0, 0)
ERROR_CORE_RECT = Rect(0.1, 0.1, 0.1, 0.1)  # anything out of range (0.1, 1)
OUTSIDE_AREA = Rect(100, 100, VLIMAGE_ONE_FACE.rect.width, VLIMAGE_ONE_FACE.rect.height)


class TestDetector(DetectTestClass):
    """
    Test of redetection.
    """

    detector: FaceDetector

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)

    def test_get_landmarks_for_redetect_one(self):
        """
        Test get and check landmark instances for re-detection of one face
        """
        for case in self.landmarksCases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                for detector in self.detectors:
                    with self.subTest(detectorType=detector.detectorType):
                        detectOne = detector.detectOne(image=VLIMAGE_ONE_FACE)
                        redetect = detector.redetectOne(
                            image=VLIMAGE_ONE_FACE,
                            detection=detectOne,
                            detect68Landmarks=case.detect68Landmarks,
                            detect5Landmarks=case.detect5Landmarks,
                        )
                        self.assertDetectionLandmarks(
                            detection=redetect, landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks
                        )

    def test_get_landmarks_for_batch_redetect(self):
        """
        Test get and check landmark instances for batch re-detection
        """
        for case in self.landmarksCases:
            with self.subTest(landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks):
                for detector in self.detectors:
                    with self.subTest(detectorType=detector.detectorType):
                        detectOne = detector.detectOne(image=VLIMAGE_ONE_FACE)
                        redetect = detector.redetect(
                            images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[detectOne.boundingBox.rect])],
                            detect68Landmarks=case.detect68Landmarks,
                            detect5Landmarks=case.detect5Landmarks,
                        )[0][0]
                        self.assertDetectionLandmarks(
                            detection=redetect, landmarks5=case.detect5Landmarks, landmarks68=case.detect68Landmarks
                        )

    def test_redetect_one_with_bbox_option(self):
        """
        Test re-detection of one face with bounding box option
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE)
                redetect = detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=detection.boundingBox.rect)
                self.assertFaceDetection(redetect, VLIMAGE_ONE_FACE)

    def test_redetect_one_with_detection_option(self):
        """
        Test re-detection of one face with detection options
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE)
                redetect = detector.redetectOne(image=VLIMAGE_ONE_FACE, detection=detection)
                self.assertFaceDetection(redetect, VLIMAGE_ONE_FACE)

    def test_batch_redetect_with_one_face(self):
        """
        Test batch re-detection with one face image
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(image=VLIMAGE_ONE_FACE)
                redetect = detector.redetect(
                    images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[detection.boundingBox.rect])]
                )[0]
                self.assertFaceDetection(redetect, VLIMAGE_ONE_FACE)

    def test_batch_redetect(self):
        """
        Test re-detection batch of images
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detectSeveral = detector.detect(images=[VLIMAGE_ONE_FACE, VLIMAGE_SEVERAL_FACE])
                redetect = detector.redetect(
                    images=[
                        ImageForRedetection(
                            image=VLIMAGE_SEVERAL_FACE, bBoxes=[face.boundingBox.rect for face in detectSeveral[1]]
                        ),
                        ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[detectSeveral[0][0].boundingBox.rect]),
                    ]
                )
                self.assertFaceDetection(redetect[0], VLIMAGE_SEVERAL_FACE)
                self.assertFaceDetection(redetect[1], VLIMAGE_ONE_FACE)
                assert 2 == len(redetect)
                assert 5 == len(redetect[0])
                assert 1 == len(redetect[1])

    def test_redetect_by_area_without_face(self):
        """
        Test re-detection by area without face
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                redetectOne = detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=Rect(0, 0, 100, 100))
                redetect = detector.redetect(
                    images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[Rect(0, 0, 100, 100)])]
                )[0][0]
                assert redetectOne is None
                assert redetect is None

    def test_redetect_one_invalid_rectangle(self):
        """
        Test re-detection of one face with an invalid rect
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=INVALID_RECT)
                if detector.detectorType.name == "FACE_DET_V3":
                    self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidRect)
                else:
                    self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidInput)

    def test_redetect_invalid_rectangle(self):
        """
        Test batch re-detection with an invalid rect
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.redetect(images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[INVALID_RECT])])
                self.assertLunaVlError(exceptionInfo, LunaVLError.UnknownError)

    def test_redetect_one_without_detection_and_bbox(self):
        """
        Test re-detection of one face without bounding box and face detection
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    detector.redetectOne(image=VLIMAGE_ONE_FACE)
                self.assertLunaVlError(exceptionInfo, LunaVLError.DetectFacesError)

    @pytest.mark.skip("core bug: Fatal error")
    def test_rect_float(self):
        """
        Test re-detection with an invalid rect
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detector.redetect(images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[ERROR_CORE_RECT])])

    def test_match_redetect_one_image(self):
        """
        Test match of values at different re-detections (redetectOne and redetect) with one image
        """
        for image in (VLIMAGE_ONE_FACE, VLIMAGE_SMALL):
            for detector in self.detectors:
                with self.subTest(detectorType=detector.detectorType):
                    if detector.detectorType.name == "FACE_DET_V3":
                        self.skipTest("Skip for FaceDetV3. Different value")
                        continue
                    bBoxRect = detector.detectOne(image=image).boundingBox.rect
                    redetectOne = detector.redetectOne(image=image, bBox=bBoxRect, detect68Landmarks=True)
                    batchRedetect = detector.redetect(
                        images=[ImageForRedetection(image=image, bBoxes=[bBoxRect])] * 3, detect68Landmarks=True
                    )
                    for redetect in batchRedetect:
                        for face in redetect:
                            assert face.boundingBox.asDict() == redetectOne.boundingBox.asDict()
                            assert face.landmarks5.asDict() == redetectOne.landmarks5.asDict()
                            assert face.landmarks68.asDict() == redetectOne.landmarks68.asDict()

    def test_redetect_one_in_area_outside_image(self):
        """
        Test re-detection of one face in area outside image
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                if detector.detectorType.name == "FACE_DET_V3":
                    redetectOne = detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=OUTSIDE_AREA)
                    self.assertFaceDetection(redetectOne, VLIMAGE_ONE_FACE)
                else:
                    redetectOne = detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=OUTSIDE_AREA)
                    assert redetectOne is None

    def test_batch_redetect_in_area_outside_image(self):
        """
        Test batch re-detection in area outside image
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                if detector.detectorType.name == "FACE_DET_V3":
                    redetect = detector.redetect(
                        images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[OUTSIDE_AREA])]
                    )
                    self.assertFaceDetection(redetect[0], VLIMAGE_ONE_FACE)
                else:
                    redetect = detector.redetect(
                        images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[OUTSIDE_AREA])]
                    )
                    assert redetect[0][0] is None
