from lunavl.sdk.estimators.image_estimators.orientation_mode import OrientationModeEstimator, OrientationType
from lunavl.sdk.image_utils.image import VLImage, ImageAngle
from tests.base import BaseTestClass
from tests.resources import ROTATED0


class TestImageRotation(BaseTestClass):
    """
    Test image rotation, using orientation mode estimator to validate expected results.
    """

    # orientation mode estimator
    orientationModeEstimator: OrientationModeEstimator

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.orientationModeEstimator = cls.faceEngine.createOrientationModeEstimator()
        cls.image = VLImage.load(filename=ROTATED0)

    def test_rotate_normal(self):
        """
        Test image rotation - 0 angle
        """
        rotatedImage = VLImage.rotate(self.image, ImageAngle.ANGLE_0)
        orientationMode = self.orientationModeEstimator.estimate(rotatedImage)
        assert orientationMode == OrientationType.NORMAL

    def test_rotate_left(self):
        """
        Test image rotation - 90 angle
        """
        rotatedImage = VLImage.rotate(self.image, ImageAngle.ANGLE_90)
        orientationMode = self.orientationModeEstimator.estimate(rotatedImage)
        assert orientationMode == OrientationType.LEFT

    def test_rotate_right(self):
        """
        Test image rotation - 270 angle
        """
        rotatedImage = VLImage.rotate(self.image, ImageAngle.ANGLE_270)
        orientationMode = self.orientationModeEstimator.estimate(rotatedImage)
        assert orientationMode == OrientationType.RIGHT

    def test_rotate_upside_down(self):
        """
        Test image rotation - 180 angle
        """
        rotatedImage = VLImage.rotate(self.image, ImageAngle.ANGLE_180)
        orientationMode = self.orientationModeEstimator.estimate(rotatedImage)
        assert orientationMode == OrientationType.UPSIDE_DOWN
