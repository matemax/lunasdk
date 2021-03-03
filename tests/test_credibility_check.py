from typing import Dict

from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.estimators.face_estimators.credibility_check import CredibilityCheck, CredibilityCheckEstimator
from lunavl.sdk.image_utils.image import VLImage

from tests.base import BaseTestClass
from tests.resources import WARP_CLEAN_FACE
from tests.schemas import jsonValidator, CREDIBILITY_CHECK_SCHEMA


class TestCredibilityCheck(BaseTestClass):
    """
    Test estimate credibility check.
    """

    credibilityCheckEstimator: CredibilityCheckEstimator

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.credibilityCheckEstimator = cls.faceEngine.createCredibilityCheckEstimator()

        cls.warp = FaceWarpedImage(VLImage.load(filename=WARP_CLEAN_FACE))

    def assertrCedibilityCheckEstimation(self, credibilityCheck: CredibilityCheck, expectedEstimationResults: float):
        """
        Function checks if the instance belongs to the credibility check class 
        and compares the result with what is expected.

        Args:
            credibilityCheck: credibility check estimation object
            expectedEstimationResults: dictionary with result
        """
        assert isinstance(credibilityCheck, CredibilityCheck), f"{credibilityCheck.__class__} is not {CredibilityCheck}"
        credibilityCheckScore = credibilityCheck.credibilityCheck
        assert isinstance(credibilityCheck.credibilityCheck, float), f"{credibilityCheckScore.__class__} is not float"
        assert 0 <= credibilityCheckScore <= 1, f"{credibilityCheckScore} not in range [0, 1]"
        self.assertAlmostEqual(
            credibilityCheckScore, expectedEstimationResults, delta=0.001, msg=f"property value is incorrect"
        )

    def test_estimate_credibility_check_as_dict(self):
        """
        Test credibility check estimations as dict
        """
        credibilityCheck = TestCredibilityCheck.credibilityCheckEstimator.estimate(self.warp).asDict()
        assert (
            jsonValidator(schema=CREDIBILITY_CHECK_SCHEMA).validate(credibilityCheck) is None
        ), f"{credibilityCheck} does not match with schema {CREDIBILITY_CHECK_SCHEMA}"

    def test_estimate(self):
        """
        Test credibility check estimations
        """
        expectedResult = 0.926
        credibilityCheck = TestCredibilityCheck.credibilityCheckEstimator.estimate(self.warp)
        self.assertrCedibilityCheckEstimation(credibilityCheck, expectedResult)
