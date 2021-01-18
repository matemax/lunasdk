from lunavl.sdk.estimators.face_estimators.orientation_mode import OrientationType, OrientationModeEstimator, \
    OrientationMode
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ROTATED0, ROTATED90, ROTATED180, ROTATED270
from tests.schemas import jsonValidator, ORIENTATION_MODE_SCHEMA


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

        cls.normalImage = VLImage.load(filename=ROTATED0).coreImage
        cls.rotatedLeftImage = VLImage.load(filename=ROTATED90).coreImage
        cls.rotatedRightImage = VLImage.load(filename=ROTATED270).coreImage
        cls.upsideDownImage = VLImage.load(filename=ROTATED180).coreImage

    @staticmethod
    def assertOrientationModeEstimation(orientationType: OrientationMode, expectedOrientation: OrientationType):
        """
        Function checks if the instance belongs to the Orientation mode class and
        compares the result with what is expected.

        Args:
            orientationType: orientation type estimation object
            expectedOrientation: expected image orientation
        """
        assert isinstance(orientationType, OrientationMode), f"{orientationType.__class__} is not {OrientationMode}"
        assert orientationType.orientationMode == expectedOrientation.value, \
            f"Expected {expectedOrientation.value}, got {orientationType.orientationMode}"

    def test_orientation_mode_as_dict(self):
        """
        Test orientation mode estimations as dict
        """
        orientationModeDict = TestOrientationMode.orientationModeEstimator.estimate(self.normalImage).asDict()
        assert (
                jsonValidator(schema=ORIENTATION_MODE_SCHEMA).validate(orientationModeDict) is None
        ), f"{orientationModeDict} does not match with schema {ORIENTATION_MODE_SCHEMA}"

    def test_orientation_mode_normal(self):
        """
        Test orientation mode with normal rotation
        """
        orientationMode = TestOrientationMode.orientationModeEstimator.estimate(self.normalImage)
        self.assertOrientationModeEstimation(orientationMode, OrientationType.NORMAL)

    def test_orientation_mode_rotated_left(self):
        """
        Test orientation mode with left rotation
        """
        orientationMode = TestOrientationMode.orientationModeEstimator.estimate(self.rotatedLeftImage)
        self.assertOrientationModeEstimation(orientationMode, OrientationType.LEFT)

    def test_orientation_mode_rotated_right(self):
        """
        Test orientation mode with right rotation
        """
        orientationMode = TestOrientationMode.orientationModeEstimator.estimate(self.rotatedRightImage)
        self.assertOrientationModeEstimation(orientationMode, OrientationType.RIGHT)

    def test_orientation_mode_rotated_upside_down(self):
        """
        Test orientation mode with upside-down rotation
        """
        orientationMode = TestOrientationMode.orientationModeEstimator.estimate(self.upsideDownImage)
        self.assertOrientationModeEstimation(orientationMode, OrientationType.UPSIDE_DOWN)
