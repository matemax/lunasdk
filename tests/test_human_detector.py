from typing import List

import pytest

from lunavl.sdk.base import BoundingBox
from lunavl.sdk.detectors.base import ImageForDetection
from lunavl.sdk.detectors.bodydetector import BodyDetection
from lunavl.sdk.detectors.facedetector import FaceDetection
from lunavl.sdk.detectors.humandetector import HumanDetector, HumanDetection
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import ColorFormat, VLImage
from tests.base import BaseTestClass
from tests.detect_test_class import (
    AREA_WITHOUT_FACE,
    GOOD_AREA,
    OUTSIDE_AREA,
    VLIMAGE_ONE_FACE,
    VLIMAGE_SEVERAL_FACE,
)
from tests.resources import (
    MANY_FACES,
    NO_FACES,
    ONE_FACE,
    IMAGE_WITH_TWO_BODY_ONE_FACE,
    WARP_FACE_WITH_SUNGLASSES,
)


class TestBodyDetector(BaseTestClass):
    """
    Test of detector.
    """

    #: global human detector
    detector: HumanDetector

    @classmethod
    def setup_class(cls):
        """
        Create list of face detector
        """
        super().setup_class()
        cls.detector = cls.faceEngine.createHumanDetector()

    def assertBoundingBox(self, boundingBox: BoundingBox):
        """
        Assert attributes of Bounding box class

        Args:
            boundingBox: bounding box
        """
        assert isinstance(boundingBox, BoundingBox), f"{boundingBox} is not {BoundingBox}"
        self.checkRectAttr(boundingBox.rect)

        assert isinstance(boundingBox.score, float), f"{boundingBox.score} is not float"
        assert 0 <= boundingBox.score < 1, "score out of range [0,1]"

    def assertDetections(
        self,
        detections: List[HumanDetection],
        imageVl: VLImage,
    ):
        """
        Function checks if an instance is Detection class, image and bounding box

        Args:
            detections: detections on image
            imageVl: class image
        """
        assert 1 <= len(detections)

        for detection in detections:
            assert isinstance(detection, HumanDetection), f"{detection.__class__} is not HumanDetection"
            assert detection.image.asPillow() == imageVl.asPillow(), "Detection image does not match VLImage"
            if body := detection.body:
                assert isinstance(body, BodyDetection)
                assert body.coreEstimation.isValid()
                self.assertBoundingBox(body.boundingBox)
                assert body.landmarks17 is None

            if face := detection.face:
                assert isinstance(face, FaceDetection)
                assert face.coreEstimation.isValid()
                assert face.landmarks5 is None
                assert face.landmarks68 is None
                self.assertBoundingBox(face.boundingBox)
            assert face or body
            assert detection.associationScore is None or (0 <= detection.associationScore <= 1)

    def test_human_detection(self):
        """
        Test structure image for detection
        """
        imageDetections = self.detector.detect(images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=GOOD_AREA)])
        assert 1 == len(imageDetections)
        detections = imageDetections[0]
        self.assertDetections(detections, VLIMAGE_ONE_FACE)
        assert 1 == len(detections)
        assert detections[0].body
        assert detections[0].face

    def test_method_as_dict(self):
        """
        Test as dict method
        """
        detection = self.detector.detect(images=[VLIMAGE_ONE_FACE])[0][0]

        assert {
            "face": detection.face.asDict(),
            "body": detection.body.asDict(),
            "association_score": detection.associationScore,
        } == detection.asDict()

    def test_batch_detect_with_success_and_error(self):
        """
        Test batch detection with success and error using FACE_DET_V3 (there is not error with other detector)
        """
        badImage = VLImage.load(filename=ONE_FACE, colorFormat=ColorFormat.B8G8R8)
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.detector.detect(images=[VLIMAGE_ONE_FACE, badImage])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        assert len(exceptionInfo.value.context) == 1, "Expect one error in exception context"
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.InvalidImageFormat)

    def test_detect_one_with_image_of_several_humans(self):
        """
        Test detection of one human with image of several humans
        """

        detections = self.detector.detect(images=[VLIMAGE_SEVERAL_FACE])[0]
        assert 5 == len(detections)
        self.assertDetections(detections, VLIMAGE_SEVERAL_FACE)

    def test_batch_detect_with_image_without_humans(self):
        """
        Test batch human detection with image without humans
        """
        imageWithoutFace = VLImage.load(filename=NO_FACES)

        detection = self.detector.detect([imageWithoutFace])
        assert detection == [[]]

    def test_detect_one_by_area_without_human(self):
        """
        Test detection of one human by area without human
        """
        detection = self.detector.detect(images=[ImageForDetection(VLIMAGE_ONE_FACE, detectArea=AREA_WITHOUT_FACE)])
        assert detection == [[]]

    def test_detect_one_by_area_with_human(self):
        """
        Test detection of one human by area with human
        """
        detections = self.detector.detect(images=[ImageForDetection(VLIMAGE_ONE_FACE, detectArea=GOOD_AREA)])[0]
        self.assertDetections(detections, VLIMAGE_ONE_FACE)

    def test_batch_detect_of_multiple_images(self):
        """
        Test batch detection of multiple images
        """
        detection = self.detector.detect(images=[VLIMAGE_SEVERAL_FACE, VLIMAGE_ONE_FACE])
        self.assertDetections(detection[0], VLIMAGE_SEVERAL_FACE)
        self.assertDetections(detection[1], VLIMAGE_ONE_FACE)
        assert 2 == len(detection)
        assert 5 == len(detection[0])
        assert 1 == len(detection[1])

    def test_batch_detect_many_faces(self):
        """
        Test checking detection limit for an image
        """
        imageWithManyFaces = VLImage.load(filename=MANY_FACES)

        detections = self.detector.detect(images=[imageWithManyFaces])[0]
        self.assertDetections(detections, imageWithManyFaces)

    def test_detect_one_invalid_image_format(self):
        """
        Test invalid image format detection
        """
        imageWithOneFaces = VLImage.load(filename=ONE_FACE, colorFormat=ColorFormat.B8G8R8)
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.detector.detect([imageWithOneFaces])
        assert len(exceptionInfo.value.context) == 1, "Expect one error in exception context"
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.InvalidImageFormat)

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

    def test_detect_body_without_face(self):
        """
        Test detect body without face
        """

        image = VLImage.load(filename=IMAGE_WITH_TWO_BODY_ONE_FACE)
        res = self.detector.detect(images=[image])
        assert 1 == len(res)
        assert 2 == len(res[0])
        detections = res[0]
        self.assertDetections(detections, image)
        detections.sort(key=lambda detection: detection.body.boundingBox.rect.x)
        assert detections[0].face is None
        assert detections[1].face is not None
        assert detections[0].body is not None
        assert detections[1].body is not None

    def test_detect_face_and_body_without_associations(self):
        """
        Test detect face and  body without associations
        """

        image = VLImage.load(filename=WARP_FACE_WITH_SUNGLASSES)
        res = self.detector.detect(images=[image])
        detections = res[0]
        self.assertDetections(detections, image)
        assert 2 == len(detections)
        assert detections[0].associationScore is None
        assert detections[1].associationScore is None

        assert detections[0].body or detections[1].body
        assert detections[0].face or detections[1].face

        if detections[0].face:
            assert detections[0].body is None
        else:
            assert detections[0].body is not None

        if detections[1].face:
            assert detections[1].body is None
        else:
            assert detections[1].body is not None

    def test_batch_detect_in_area_outside_image(self):
        """
        Test batch detection in area outside image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.detector.detect(images=[ImageForDetection(image=VLIMAGE_ONE_FACE, detectArea=OUTSIDE_AREA)])

        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        assert len(exceptionInfo.value.context) == 1, "Expect one error in exception context"
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.InvalidRect)

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

    def test_async_detect_human(self):
        """
        Test async detect human
        """
        task = self.detector.detect([VLIMAGE_ONE_FACE] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, BodyDetection)
