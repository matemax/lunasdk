from collections import namedtuple
from typing import List

import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.image_estimators.orientation_mode import OrientationModeEstimator, OrientationType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ROTATED0, ROTATED90, ROTATED180, ROTATED270

ImageNExpectedOrientationMode = namedtuple("ImageNExpectedOrientationMode", ("image", "expectedOrientationMode"))


class TestOrientationMode(BaseTestClass):
    """
    Test estimate orientation mode.
    """

    # orientation mode estimator
    orientationModeEstimator: OrientationModeEstimator
    # images with expected orientation modes
    testData: List[ImageNExpectedOrientationMode]

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.orientationModeEstimator = cls.faceEngine.createOrientationModeEstimator()
        cls.testData = [
            ImageNExpectedOrientationMode(VLImage.load(filename=ROTATED0), OrientationType.NORMAL),
            ImageNExpectedOrientationMode(VLImage.load(filename=ROTATED90), OrientationType.LEFT),
            ImageNExpectedOrientationMode(VLImage.load(filename=ROTATED270), OrientationType.RIGHT),
            ImageNExpectedOrientationMode(VLImage.load(filename=ROTATED180), OrientationType.UPSIDE_DOWN),
        ]

    @staticmethod
    def assertOrientationModeEstimation(estimatedOrientation: OrientationType, expectedOrientation: OrientationType):
        """
        Function checks if the instance belongs to the Orientation mode class and
        compares the result with what is expected.

        Args:
            estimatedOrientation: orientation estimation
            expectedOrientation: expected image orientation
        """
        assert isinstance(
            estimatedOrientation, OrientationType
        ), f"{estimatedOrientation.__class__} is not {OrientationType}"
        assert (
            estimatedOrientation == expectedOrientation
        ), f"Expected {expectedOrientation.value}, got {estimatedOrientation}"

    def test_orientation_mode(self):
        """
        Test orientation mode with normal, left, right and upside-down rotation
        """
        for image, expectedOrientationType in self.testData:
            with self.subTest(expectedOrientationType=expectedOrientationType):
                orientationMode = TestOrientationMode.orientationModeEstimator.estimate(image)
                self.assertOrientationModeEstimation(orientationMode, expectedOrientationType)

    def test_orientation_mode_batch(self):
        """
        Test orientation mode with two images
        """
        images = [image.image for image in self.testData]
        orientationModeList = self.orientationModeEstimator.estimateBatch(images)
        assert isinstance(orientationModeList, list)
        assert len(orientationModeList) == len(images)
        for idx, orientationMode in enumerate(orientationModeList):
            self.assertOrientationModeEstimation(orientationMode, self.testData[idx].expectedOrientationMode)

    def test_orientation_mode_batch_invalid_input(self):
        """
        Test orientation mode batch with invalid input
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.orientationModeEstimator.estimateBatch([])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidSpanSize)
