from lunavl.sdk.estimators.image_estimators.orientation_mode import OrientationModeEstimator, OrientationType
from lunavl.sdk.image_utils.image import VLImage, ImageAngle
from tests.base import BaseTestClass
from tests.resources import ROTATED0, ROTATED90, ROTATED180, ROTATED270


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

    def test_image_rotation(self):
        """
        Test image rotation: 0, 90, 180 and 270 degrees
        """
        testData = [
            (ImageAngle.ANGLE_0, OrientationType.NORMAL, ROTATED0),
            (ImageAngle.ANGLE_90, OrientationType.LEFT, ROTATED90),
            (ImageAngle.ANGLE_180, OrientationType.UPSIDE_DOWN, ROTATED180),
            (ImageAngle.ANGLE_270, OrientationType.RIGHT, ROTATED270),
        ]

        for rotationAngle, expectedOrientationMode, expectedImageFileName in testData:
            with self.subTest(rotationAngle=rotationAngle):
                rotatedImage = VLImage.rotate(self.image, rotationAngle)
                orientationMode = self.orientationModeEstimator.estimate(rotatedImage)
                assert orientationMode == expectedOrientationMode
                assert (
                    VLImage.load(filename=expectedImageFileName).asPillow().tobytes()
                    == rotatedImage.asPillow().tobytes()
                )
