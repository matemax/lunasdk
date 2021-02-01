from lunavl.sdk.estimators.image_estimators.orientation_mode import OrientationModeEstimator, OrientationType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ROTATED0, ROTATED90, ROTATED180, ROTATED270


class TestOrientationMode(BaseTestClass):
    """
    Test estimate orientation mode.
    """

    # orientation mode estimator
    orientationModeEstimator: OrientationModeEstimator

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.orientationModeEstimator = cls.faceEngine.createOrientationModeEstimator()

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
        testData = [
            (VLImage.load(filename=ROTATED0), OrientationType.NORMAL),
            (VLImage.load(filename=ROTATED90), OrientationType.LEFT),
            (VLImage.load(filename=ROTATED270), OrientationType.RIGHT),
            (VLImage.load(filename=ROTATED180), OrientationType.UPSIDE_DOWN),
        ]

        for image, expectedOrientationType in testData:
            with self.subTest(expectedOrientationType=expectedOrientationType):
                orientationMode = TestOrientationMode.orientationModeEstimator.estimate(image)
                self.assertOrientationModeEstimation(orientationMode, expectedOrientationType)
