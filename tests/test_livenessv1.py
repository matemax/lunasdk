from collections import namedtuple
from typing import Optional

import pytest

from lunavl.sdk.estimators.face_estimators.livenessv1 import LivenessPrediction, LivenessV1
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ONE_FACE, SPOOF, UNKNOWN_LIVENESS, LIVENESS_FACE, SMALL_IMAGE
from tests.schemas import LIVENESSV1_SCHEMA, jsonValidator


class TestEstimateLivenessV1(BaseTestClass):
    """
    Test liveness estimation.
    """

    @classmethod
    def setup_class(cls):
        """
        Create test data and estimators
        """
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)
        cls.headPoseEstimator = cls.faceEngine.createHeadPoseEstimator()
        cls.livenessEstimator = cls.faceEngine.createLivenessV1Estimator()
        cls.detection = cls.detector.detectOne(VLImage.load(filename=ONE_FACE), detect68Landmarks=True)

    def assertLivenessEstimation(self, estimation: LivenessV1, expectedPrediction: Optional[LivenessPrediction] = None):
        """
        Assert estimation liveness.

        Args:
            estimation: estimation
            expectedPrediction: expected prediction
        """
        assert isinstance(estimation, LivenessV1), f"wrong estimation type {type(estimation)}"
        assert isinstance(
            estimation.prediction, LivenessPrediction
        ), f"wrong estimation type {type(estimation.prediction)}"
        assert isinstance(estimation.quality, float)
        assert isinstance(estimation.score, float)
        if expectedPrediction:
            assert expectedPrediction == estimation.prediction, (
                f"wrong prediction {estimation.prediction}, " f"expected {expectedPrediction}"
            )

    def test_livenessv1_as_dict(self):
        """
        Test liveness estimations as dict
        """
        livenessDict = self.livenessEstimator.estimate(self.detection).asDict()
        assert (
            jsonValidator(schema=LIVENESSV1_SCHEMA).validate(livenessDict) is None
        ), f"{livenessDict} does not match with schema {LIVENESSV1_SCHEMA}"

    def test_liveness_estimation(self):
        """
        Test liveness estimator
        """
        Case = namedtuple("Case", ("image", "prediction"))
        cases = (
            Case(LIVENESS_FACE, LivenessPrediction.Real),
            Case(SPOOF, LivenessPrediction.Spoof),
            Case(UNKNOWN_LIVENESS, LivenessPrediction.Unknown),
        )
        for case in cases:
            with self.subTest(prediction=case.prediction):
                faceDetection = self.detector.detectOne(VLImage.load(filename=case.image), detect68Landmarks=True)
                estimation = self.livenessEstimator.estimate(faceDetection=faceDetection, qualityThreshold=0.9)
                self.assertLivenessEstimation(estimation, expectedPrediction=case.prediction)

    def test_liveness_estimation_quality_threshold(self):
        """
        Test a quality threshold of liveness estimator
        """
        estimation = self.livenessEstimator.estimate(faceDetection=self.detection)
        self.assertLivenessEstimation(estimation, expectedPrediction=LivenessPrediction.Real)
        estimation = self.livenessEstimator.estimate(
            faceDetection=self.detection, qualityThreshold=estimation.quality + 0.001
        )
        self.assertLivenessEstimation(estimation, expectedPrediction=LivenessPrediction.Unknown)
        estimation = self.livenessEstimator.estimate(
            faceDetection=self.detection, qualityThreshold=estimation.quality - 0.001
        )
        self.assertLivenessEstimation(estimation, expectedPrediction=LivenessPrediction.Real)

    def test_estimate_liveness_by_detection_without_landmarks5(self):
        """
        Test estimate liveness by face detection without landmarks5. Todo: remove after FSDK-2811
        """
        faceDetection = self.detector.detectOne(VLImage.load(filename=SMALL_IMAGE), detect5Landmarks=False)
        with pytest.raises(ValueError) as exceptionInfo:
            self.livenessEstimator.estimate(faceDetection=faceDetection)
        assert "Landmarks5 is required for liveness estimation" == str(exceptionInfo.value)
