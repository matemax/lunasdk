from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.estimators.face_estimators.credibility import Credibility, CredibilityEstimator
from lunavl.sdk.image_utils.image import VLImage

from tests.base import BaseTestClass
from tests.resources import WARP_CLEAN_FACE
from tests.schemas import jsonValidator, CREDIBILITY_SCHEMA


class TestCredibility(BaseTestClass):
    """
    Test estimate credibility.
    """

    credibilityEstimator: CredibilityEstimator

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.credibilityEstimator = cls.faceEngine.createCredibilityEstimator()

        cls.warp = FaceWarpedImage(VLImage.load(filename=WARP_CLEAN_FACE))

    def assertCredibilityEstimation(self, credibility: Credibility, expectedEstimationResults: float):
        """
        Function checks if the instance belongs to the credibility class
        and compares the result with what is expected.

        Args:
            credibility: credibility estimation object
            expectedEstimationResults: dictionary with result
        """
        assert isinstance(credibility, Credibility), f"{credibility.__class__} is not {Credibility}"
        score = credibility.score
        assert isinstance(score, float), f"{credibility.__class__} is not float"
        assert 0 <= score <= 1, f"{score} not in range [0, 1]"
        self.assertAlmostEqual(score, expectedEstimationResults, delta=0.001, msg="property value is incorrect")

    def test_estimate_credibility_as_dict(self):
        """
        Test credibility credibility as dict
        """
        credibilityCheck = TestCredibility.credibilityEstimator.estimate(self.warp).asDict()
        assert (
            jsonValidator(schema=CREDIBILITY_SCHEMA).validate(credibilityCheck) is None
        ), f"{credibilityCheck} does not match with schema {CREDIBILITY_SCHEMA}"

    def test_estimate(self):
        """
        Test credibility check estimations
        """
        expectedResult = 0.923
        credibility = TestCredibility.credibilityEstimator.estimate(self.warp)
        self.assertCredibilityEstimation(credibility, expectedResult)

    def test_async_estimation(self):
        """
        Test async credibility check estimations
        """
        task = self.credibilityEstimator.estimate(self.warp, asyncEstimate=True)
        self.assertAsyncEstimation(task, Credibility)
        self.assertCredibilityEstimation(task.get(), 0.923)
