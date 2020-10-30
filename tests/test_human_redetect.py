import pytest

from lunavl.sdk.detectors.base import ImageForRedetection
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.image_utils.geometry import Rect
from tests.detect_test_class import HumanDetectTestClass
from tests.detect_test_class import (
    VLIMAGE_ONE_FACE,
    VLIMAGE_SEVERAL_FACE,
    VLIMAGE_SMALL,
    OUTSIDE_AREA,
    INVALID_RECT,
    ERROR_CORE_RECT,
)


class TestsRedetectHuman(HumanDetectTestClass):
    """
    Human redetection tests.
    """

    def test_redetect_one_with_bbox_option(self):
        """
        Test re-detection of one human with bounding box option
        """

        detection = self.detector.detectOne(image=VLIMAGE_ONE_FACE)
        redetect = self.detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=detection.boundingBox.rect)
        self.assertHumanDetection(redetect, VLIMAGE_ONE_FACE)

    def test_redetect_one_with_detection_option(self):
        """
        Test re-detection of one human with detection options
        """
        detection = self.detector.detectOne(image=VLIMAGE_ONE_FACE)
        redetect = self.detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=detection)
        self.assertHumanDetection(redetect, VLIMAGE_ONE_FACE)

    def test_batch_redetect_with_one_human(self):
        """
        Test batch re-detection with one human image
        """
        detection = self.detector.detectOne(image=VLIMAGE_ONE_FACE)
        redetect = self.detector.redetect(
            images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[detection.boundingBox.rect])]
        )[0]
        self.assertHumanDetection(redetect, VLIMAGE_ONE_FACE)

    def test_batch_redetect(self):
        """
        Test re-detection batch of images
        """
        detectSeveral = self.detector.detect(images=[VLIMAGE_ONE_FACE, VLIMAGE_SEVERAL_FACE])
        redetect = self.detector.redetect(
            images=[
                ImageForRedetection(
                    image=VLIMAGE_SEVERAL_FACE, bBoxes=[human.boundingBox.rect for human in detectSeveral[1]]
                ),
                ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[detectSeveral[0][0].boundingBox.rect]),
            ]
        )

        assert 2 == len(redetect)
        self.assertHumanDetection(redetect[0], VLIMAGE_SEVERAL_FACE)
        self.assertHumanDetection(redetect[1], VLIMAGE_ONE_FACE)
        assert 5 == len(redetect[0])
        assert 1 == len(redetect[1])

    def test_redetect_by_area_without_human(self):
        """
        Test re-detection by area without human
        """
        redetectOne = self.detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=Rect(0, 0, 100, 100))
        redetect = self.detector.redetect(
            images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[Rect(0, 0, 100, 100)])]
        )[0][0]
        assert redetectOne is None, "excepted None but found {}".format(redetectOne)
        assert redetect is None, "excepted None but found {}".format(redetectOne)

    def test_redetect_one_invalid_rectangle(self):
        """
        Test re-detection of one human with an invalid rect
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=INVALID_RECT)
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidRect)

    @pytest.mark.skip()  # TODO get resolution from n.feofanov
    def test_redetect_invalid_rectangle(self):
        """
        Test batch re-detection with an invalid rect
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.detector.redetect(
                images=[
                    ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[INVALID_RECT]),
                    ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[Rect(0, 0, 100, 100)]),
                ]
            )
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError)
        assert len(exceptionInfo.value.context) == 1, "Expect one error in exception context"
        assert exceptionInfo.value.context[0], LunaVLError.InvalidRect

    @pytest.mark.skip("core bug: Fatal error")
    def test_rect_float(self):
        """
        Test re-detection with an invalid rect
        """
        self.detector.redetect(images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[ERROR_CORE_RECT])])

    def test_match_redetect_one_image(self):
        """
        Test match of values at different re-detections (redetectOne and redetect) with one image
        """
        for image in (VLIMAGE_ONE_FACE, VLIMAGE_SMALL):
            bBoxRect = self.detector.detectOne(image=image).boundingBox.rect
            redetectOne = self.detector.redetectOne(image=image, bBox=bBoxRect)
            batchRedetect = self.detector.redetect(images=[ImageForRedetection(image=image, bBoxes=[bBoxRect])] * 3)
            for redetect in batchRedetect:
                for human in redetect:
                    assert human.boundingBox.asDict() == redetectOne.boundingBox.asDict()

    def test_redetect_one_in_area_outside_image(self):
        """
        Test re-detection of one human in area outside image
        """
        redetectOne = self.detector.redetectOne(image=VLIMAGE_ONE_FACE, bBox=OUTSIDE_AREA)
        self.assertHumanDetection(redetectOne, VLIMAGE_ONE_FACE)

    def test_batch_redetect_in_area_outside_image(self):
        """
        Test batch re-detection in area outside image
        """
        redetect = self.detector.redetect(images=[ImageForRedetection(image=VLIMAGE_ONE_FACE, bBoxes=[OUTSIDE_AREA])])
        self.assertHumanDetection(redetect[0], VLIMAGE_ONE_FACE)
