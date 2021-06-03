from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.estimators.face_estimators.trustworthiness import Trustworthiness, TrustworthinessEstimator
from lunavl.sdk.image_utils.image import VLImage

from tests.base import BaseTestClass
from tests.resources import WARP_CLEAN_FACE
from tests.schemas import jsonValidator, TRUSTWORTHINESS_SCHEMA


class TestCredibilityCheck(BaseTestClass):
    """
    Test estimate trustworthiness.
    """

    trustworthinessEstimator: TrustworthinessEstimator

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.trustworthinessEstimator = cls.faceEngine.createTrustworthinessEstimator()

        cls.warp = FaceWarpedImage(VLImage.load(filename=WARP_CLEAN_FACE))

    def assertTrustworthinessEstimation(self, trustworthiness: Trustworthiness, expectedEstimationResults: float):
        """
        Function checks if the instance belongs to the trustworthiness class
        and compares the result with what is expected.

        Args:
            trustworthiness: trustworthiness estimation object
            expectedEstimationResults: dictionary with result
        """
        assert isinstance(trustworthiness, Trustworthiness), f"{trustworthiness.__class__} is not {Trustworthiness}"
        credibilityCheckScore = trustworthiness.trustworthiness
        assert isinstance(trustworthiness.trustworthiness, float), f"{credibilityCheckScore.__class__} is not float"
        assert 0 <= credibilityCheckScore <= 1, f"{credibilityCheckScore} not in range [0, 1]"
        self.assertAlmostEqual(
            credibilityCheckScore, expectedEstimationResults, delta=0.001, msg="property value is incorrect"
        )

    def test_estimate_credibility_check_as_dict(self):
        """
        Test credibility check estimations as dict
        """
        credibilityCheck = TestCredibilityCheck.trustworthinessEstimator.estimate(self.warp).asDict()
        assert (
            jsonValidator(schema=TRUSTWORTHINESS_SCHEMA).validate(credibilityCheck) is None
        ), f"{credibilityCheck} does not match with schema {TRUSTWORTHINESS_SCHEMA}"

    def test_estimate(self):
        """
        Test credibility check estimations
        """
        expectedResult = 0.926
        trustworthiness = TestCredibilityCheck.trustworthinessEstimator.estimate(self.warp)
        self.assertTrustworthinessEstimation(trustworthiness, expectedResult)
