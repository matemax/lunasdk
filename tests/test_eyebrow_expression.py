from lunavl.sdk.estimators.face_estimators.eyebrow_expressions import EyebrowExpressions, EyebrowExpression
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import (
    ONE_FACE,
    SQUINTING,
    FROWNING,
    RAISED,
)


class TestEyeybrowExpression(BaseTestClass):
    """
    Test estimate eyebrow expression.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.headwearEstimator = cls.faceEngine.createEyebrowExpressionEstimator()

    def test_estimate_eyebrow_expression_correctness(self):
        """
        Test eyebrow expression estimator correctness
        """
        images = {
            EyebrowExpression.Neutral: ONE_FACE,
            EyebrowExpression.Squinting: SQUINTING,
            EyebrowExpression.Raised: RAISED,
            EyebrowExpression.Frowning: FROWNING,
        }
        for eyebrowExpression, image in images.items():
            with self.subTest(eyebrowExpression):
                estimation = self.estimate(image)
                assert eyebrowExpression == estimation.predominateExpression
                assert estimation.asDict()["predominant_expression"] == eyebrowExpression.name.lower()

    def estimate(self, image: str = ONE_FACE) -> EyebrowExpressions:
        """Estimate eyebrow expressions on image"""
        faceDetection = self.detector.detectOne(VLImage.load(filename=image))
        warp = self.warper.warp(faceDetection)
        estimation = self.headwearEstimator.estimate(warp)
        assert isinstance(estimation, EyebrowExpressions)
        return estimation

    def test_estimate_eyebrow_expression(self):
        """
        Simple headwear estimation
        """
        estimation = self.estimate(ONE_FACE)
        assert estimation.predominateExpression == EyebrowExpression.Neutral
        assert 0 <= estimation.neutral <= 1
        assert 0 <= estimation.squinting <= 1
        assert 0 <= estimation.frowning <= 1
        assert 0 <= estimation.raised <= 1

    def test_eyebrow_expression_as_dict(self):
        """
        Test method EyebrowExpression.asDict
        """
        estimation = self.estimate(ONE_FACE)
        assert {
            "predominant_expression": "neutral",
            "estimations": {
                "neutral": estimation.neutral,
                "raised": estimation.raised,
                "squinting": estimation.squinting,
                "frowning": estimation.frowning,
            },
        } == estimation.asDict()

    def test_estimate_eyebrow_expression_batch(self):
        """
        Batch eyebrow expression estimation test
        """
        faceDetections = self.detector.detect([VLImage.load(filename=ONE_FACE), VLImage.load(filename=RAISED)])
        warp1 = self.warper.warp(faceDetections[0][0])
        warp2 = self.warper.warp(faceDetections[1][0])
        estimations = self.headwearEstimator.estimateBatch([warp1, warp2])
        assert EyebrowExpression.Neutral == estimations[0].predominateExpression
        assert EyebrowExpression.Raised == estimations[1].predominateExpression
