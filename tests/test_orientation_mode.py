from FaceEngine import OrientationType

from lunavl.sdk.estimators.image_estimators.orientation_mode import OrientationModeEstimator
from lunavl.sdk.image_utils.image import VLImage, ImageAngle
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

        cls.normalImage = VLImage.load(filename=ROTATED0)
        cls.rotatedLeftImage = VLImage.load(filename=ROTATED90)
        cls.rotatedRightImage = VLImage.load(filename=ROTATED270)
        cls.upsideDownImage = VLImage.load(filename=ROTATED180)

    @staticmethod
    def assertOrientationModeEstimation(estimatedOrientation: OrientationType, expectedOrientation: ImageAngle):
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
            estimatedOrientation.name == expectedOrientation.value
        ), f"Expected {expectedOrientation.value}, got {estimatedOrientation}"

    def test_orientation_mode_normal(self):
        """
        Test orientation mode with normal rotation
        """
        orientationMode = TestOrientationMode.orientationModeEstimator.estimate(self.normalImage)
        self.assertOrientationModeEstimation(orientationMode, ImageAngle.ANGLE_0)

    def test_orientation_mode_rotated_left(self):
        """
        Test orientation mode with left rotation
        """
        orientationMode = TestOrientationMode.orientationModeEstimator.estimate(self.rotatedLeftImage)
        self.assertOrientationModeEstimation(orientationMode, ImageAngle.ANGLE_90)

    def test_orientation_mode_rotated_right(self):
        """
        Test orientation mode with right rotation
        """
        orientationMode = TestOrientationMode.orientationModeEstimator.estimate(self.rotatedRightImage)
        self.assertOrientationModeEstimation(orientationMode, ImageAngle.ANGLE_270)

    def test_orientation_mode_rotated_upside_down(self):
        """
        Test orientation mode with upside-down rotation
        """
        orientationMode = TestOrientationMode.orientationModeEstimator.estimate(self.upsideDownImage)
        self.assertOrientationModeEstimation(orientationMode, ImageAngle.ANGLE_180)
