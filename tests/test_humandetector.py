import pytest

from lunavl.sdk.detectors.base import ImageForDetection
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage, ColorFormat
from tests.detect_test_class import HumanDetectTestClass
from tests.detect_test_class import (
    VLIMAGE_ONE_FACE,
    GOOD_AREA,
    VLIMAGE_SEVERAL_FACE,
    AREA_WITHOUT_FACE,
    OUTSIDE_AREA,
    VLIMAGE_SMALL,
    VLIMAGE_BAD_IMAGE,
)
from tests.resources import ONE_FACE, MANY_FACES, NO_FACES
from tests.schemas import jsonValidator, REQUIRED_HUMAN_BODY_DETECTION, LANDMARKS17


class TestHumanDetector(HumanDetectTestClass):
    """
    Test of detector.
    """

    def test_image_detection_with_transfer_option(self):
        """
        Test structure image for detection
        """
        detection = self.detector.detect(images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=GOOD_AREA)])
        self.assertHumanDetection(detection[0], VLIMAGE_ONE_FACE)
        assert 1 == len(detection)

    def test_check_landmarks_points(self):
        """
        Test validation landmarks points
        """
        detection = self.detector.detectOne(image=VLIMAGE_ONE_FACE, detectLandmarks=True)
        self.assertHumanDetection(detection, VLIMAGE_ONE_FACE)

        self.assertLandmarksPoints(detection.landmarks17.points)

    def test_landmarks_as_dict(self):
        """
        Test conversion landmarks to dictionary
        """
        currentLandmarks17 = self.detector.detectOne(image=VLIMAGE_ONE_FACE).landmarks17.asDict()

        assert (
            jsonValidator(schema=LANDMARKS17).validate(currentLandmarks17) is None
        ), f"{currentLandmarks17} does not match with schema {LANDMARKS17}"

    def test_valid_bounding_box(self):
        """
        Test validate bounding box (rect and score)
        """
        detection = self.detector.detectOne(image=VLIMAGE_ONE_FACE)
        self.assertBoundingBox(detection.boundingBox)
        detection = self.detector.detect(images=[VLIMAGE_ONE_FACE])[0][0]
        self.assertBoundingBox(detection.boundingBox)

    def test_bounding_box_as_dict(self):
        """
        Test conversion bounding box to dictionary
        """
        boundingBox = self.detector.detectOne(image=VLIMAGE_ONE_FACE).boundingBox.asDict()

        assert (
            jsonValidator(schema=REQUIRED_HUMAN_BODY_DETECTION).validate(boundingBox) is None
        ), f"{boundingBox} does not match with schema {REQUIRED_HUMAN_BODY_DETECTION}"

    def test_human_detection_as_dict(self):
        """
        Test conversion result human detection to dictionary
        """
        for case in self.landmarksCases:
            with self.subTest(landmarks5=case.detectLandmarks):
                detectAsDict = self.detector.detectOne(
                    image=VLIMAGE_ONE_FACE, detectLandmarks=case.detectLandmarks
                ).asDict()
                assert (
                    jsonValidator(schema=REQUIRED_HUMAN_BODY_DETECTION).validate(detectAsDict) is None
                ), f"{detectAsDict} does not match with schema {REQUIRED_HUMAN_BODY_DETECTION}"

    def test_detection_with_default_detector_type(self):
        """
        Test human detection with default detector type
        """
        for detectionFunction in ("detect", "detectOne"):
            with self.subTest(detectionFunction=detectionFunction):
                if detectionFunction == "detectOne":
                    detection = self.detector.detectOne(image=VLIMAGE_ONE_FACE)
                else:
                    detection = self.detector.detect(images=[VLIMAGE_ONE_FACE])[0]
                self.assertHumanDetection(detection, VLIMAGE_ONE_FACE)

    def test_batch_detect_using_different_type_detector(self):
        """
        Test batch detection using different type of detector
        """
        detection = self.detector.detect(images=[VLIMAGE_ONE_FACE])[0]
        self.assertHumanDetection(detection, VLIMAGE_ONE_FACE)

    def test_batch_detect_with_success_and_error(self):
        """
        Test batch detection with success and error using FACE_DET_V3 (there is not error with other detector)
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.detector.detect(images=[VLIMAGE_ONE_FACE, VLIMAGE_BAD_IMAGE])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        assert len(exceptionInfo.value.context) == 1, "Expect one error in exception context"
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.InvalidRect)  # todo: ?

    def test_detect_one_with_image_of_several_humans(self):
        """
        Test detection of one human with image of several humans
        """

        detection = self.detector.detectOne(image=VLIMAGE_SEVERAL_FACE)
        self.assertHumanDetection(detection, VLIMAGE_SEVERAL_FACE)

    def test_detect_one_with_image_without_humans(self):
        """
        Test detection of one human with image without humans
        """
        imageWithoutFace = VLImage.load(filename=NO_FACES)

        detection = self.detector.detectOne(image=imageWithoutFace)
        assert detection is None, detection

    def test_batch_detect_with_image_without_humans(self):
        """
        Test batch human detection with image without humans
        """
        imageWithoutFace = VLImage.load(filename=NO_FACES)
        detection = self.detector.detect(images=[imageWithoutFace])
        assert 0 == len(detection[0])

    def test_detect_one_by_area_without_human(self):
        """
        Test detection of one human by area without human
        """
        detection = self.detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=AREA_WITHOUT_FACE)
        assert detection is None, detection

    def test_detect_one_by_area_with_human(self):
        """
        Test detection of one human by area with human
        """
        detection = self.detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=GOOD_AREA)
        self.assertHumanDetection(detection, VLIMAGE_ONE_FACE)

    def test_batch_detect_with_image_of_several_humans(self):
        """
        Test batch human detection with image of several humans
        """
        detection = self.detector.detect(images=[VLIMAGE_SEVERAL_FACE])
        self.assertHumanDetection(detection[0], VLIMAGE_SEVERAL_FACE)
        assert 1 == len(detection)
        assert 2 == len(detection[0]), f"Expected 2 faces, got {len(detection[0])}"

    def test_batch_detect_of_multiple_images(self):
        """
        Test batch detection of multiple images
        """
        detection = self.detector.detect(images=[VLIMAGE_SEVERAL_FACE, VLIMAGE_ONE_FACE])
        self.assertHumanDetection(detection[0], VLIMAGE_SEVERAL_FACE)
        self.assertHumanDetection(detection[1], VLIMAGE_ONE_FACE)
        assert 2 == len(detection)
        assert 2 == len(detection[0])
        assert 1 == len(detection[1])

    def test_get_landmarks_for_detect_one(self):
        """
        Test get and check landmark instances for detection of one human
        """
        for case in self.landmarksCases:
            with self.subTest(detect=case.detectLandmarks):
                detection = self.detector.detectOne(image=VLIMAGE_ONE_FACE, detectLandmarks=case.detectLandmarks)
                self.assertDetectionLandmarks(detection=detection, landmarksIsExpected=case.detectLandmarks)

    def test_get_landmarks_for_batch_detect(self):
        """
        Test get and check landmark instances for batch detect
        """
        for case in self.landmarksCases:
            with self.subTest(detect=case.detectLandmarks):
                detection = self.detector.detect(images=[VLIMAGE_ONE_FACE], detectLandmarks=case.detectLandmarks)[0][0]
                self.assertDetectionLandmarks(detection=detection, landmarksIsExpected=case.detectLandmarks)

    def test_batch_detect_limit(self):
        """
        Test checking detection limit for an image
        """
        imageWithManyFaces = VLImage.load(filename=MANY_FACES)

        detection = self.detector.detect(images=[imageWithManyFaces], limit=6)[0]
        assert 6 == len(detection)

        detection = self.detector.detect(images=[imageWithManyFaces], limit=20)[0]
        assert 17 == len(detection)

    @pytest.mark.skip("core bug: Fatal error")
    def test_detect_limit_bad_param(self):
        """
        Test batch detection with negative limit number
        """
        imageWithManyFaces = VLImage.load(filename=MANY_FACES)
        self.detector.detect(images=[ImageForDetection(image=imageWithManyFaces, detectArea=GOOD_AREA)], limit=-1)

    def test_detect_one_invalid_image_format(self):
        """
        Test invalid image format detection
        """
        imageWithOneFaces = VLImage.load(filename=ONE_FACE, colorFormat=ColorFormat.B8G8R8)
        errorDetail = "Bad image format for detection, format: B8G8R8, image: one_face.jpg"
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.detector.detectOne(image=imageWithOneFaces)
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidImageFormat.format(errorDetail))

    def test_batch_detect_invalid_image_format(self):
        """
        Test invalid image format detection
        """
        colorToImageMap = self.getColorToImageMap()
        allowedColorsForDetection = {ColorFormat.R8G8B8}
        for colorFormat in set(colorToImageMap) - allowedColorsForDetection:
            colorImage = colorToImageMap[colorFormat]
            with pytest.raises(LunaSDKException) as exceptionInfo:
                self.detector.detect(images=[colorImage])
            self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
            assert len(exceptionInfo.value.context) == 1, "Expect one error in exception context"
            self.assertReceivedAndRawExpectedErrors(
                exceptionInfo.value.context[0], LunaVLError.InvalidImageFormat.format("Failed validation.")
            )

    def test_batch_detect_by_area_without_human(self):
        """
        Test batch human detection by area without human
        """
        detection = self.detector.detect(
            images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=AREA_WITHOUT_FACE)]
        )
        assert 1 == len(detection)
        assert 0 == len(detection[0])

    def test_batch_detect_by_area_with_human(self):
        """
        Test batch human detection by area with human
        """
        detection = self.detector.detect(images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=GOOD_AREA)])
        assert 1 == len(detection[0])
        self.assertHumanDetection(detection[0], VLIMAGE_ONE_FACE)

    def test_bad_area_detection(self):
        """
        Test detection of one human in area outside image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=OUTSIDE_AREA)
        self.assertLunaVlError(exceptionInfo, LunaVLError.ValidationFailed.format("Failed validation."))

    def test_batch_detect_in_area_outside_image(self):
        """
        Test batch detection in area outside image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.detector.detect(images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=OUTSIDE_AREA)])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        assert len(exceptionInfo.value.context) == 1, "Expect one error in exception context"
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.InvalidRect)

    @pytest.mark.skip("unstable")
    def test_excessive_image_list_detection(self):
        """
        Test excessive image list detection
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            TestHumanDetector.defaultDetector.detect(images=[VLIMAGE_ONE_FACE] * 1000)
        self.assertLunaVlError(exceptionInfo, LunaVLError.HighMemoryUsage)

    def test_detect_one_invalid_rectangle(self):
        """
        Test detection of one human with an invalid rect
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.detector.detectOne(image=VLIMAGE_ONE_FACE, detectArea=Rect())
        self.assertLunaVlError(exceptionInfo, LunaVLError.ValidationFailed.format("Failed validation."))

    def test_batch_detect_invalid_rectangle(self):
        """
        Test batch human detection with an invalid rect
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.detector.detect(images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=Rect())])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        assert len(exceptionInfo.value.context) == 1, "Expect one error in exception context"
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[0], LunaVLError.InvalidRect.format("Failed validation.")
        )

    def test_match_detection_one_image(self):
        """
        Test match of values at different detections (detectOne and detect) with one image
        """
        for image in (VLIMAGE_ONE_FACE, VLIMAGE_SMALL):
            detectOne = self.detector.detectOne(image=image, detectLandmarks=True)
            batchDetect = self.detector.detect(images=[image] * 3, detectLandmarks=True)
            for detection in batchDetect:
                for human in detection:
                    assert human.boundingBox.asDict() == detectOne.boundingBox.asDict()
                    assert human.landmarks17.asDict() == detectOne.landmarks17.asDict()
