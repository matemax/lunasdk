from collections import namedtuple
from typing import Optional

import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.livenessv1 import LivenessPrediction, LivenessV1
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import SPOOF, UNKNOWN_LIVENESS, LIVENESS_FACE, CLEAN_ONE_FACE
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
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.headPoseEstimator = cls.faceEngine.createHeadPoseEstimator()
        cls.livenessEstimator = cls.faceEngine.createLivenessV1Estimator()
        cls.detection = cls.detector.detectOne(VLImage.load(filename=CLEAN_ONE_FACE), detect68Landmarks=True)

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
                estimation = self.livenessEstimator.estimate(faceDetection=faceDetection, qualityThreshold=0.75)
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

    def test_estimate_liveness_batch(self):
        """
        Test estimate liveness batch
        """
        detection = self.detector.detectOne(VLImage.load(filename=SPOOF), detect68Landmarks=True)
        estimations = self.livenessEstimator.estimateBatch([self.detection, detection])
        assert isinstance(estimations, list)
        assert len(estimations) == 2
        for estimation in estimations:
            self.assertLivenessEstimation(estimation)

    def test_estimate_liveness_batch_without_landmarks5(self):
        """
        Test estimate liveness batch without landmarks5
        """
        detection = self.detector.detectOne(VLImage.load(filename=SPOOF), detect5Landmarks=False)
        with pytest.raises(ValueError) as exceptionInfo:
            self.livenessEstimator.estimateBatch([detection])
        assert "Landmarks5 is required for liveness estimation" == str(exceptionInfo.value)

    def test_estimate_liveness_batch_without_landmarks68(self):
        """
        Test estimate liveness batch without landmarks 68
        """
        detection = self.detector.detectOne(VLImage.load(filename=SPOOF), detect68Landmarks=False)
        estimations = self.livenessEstimator.estimateBatch([self.detection, detection])
        assert isinstance(estimations, list)
        assert len(estimations) == 2
        for estimation in estimations:
            self.assertLivenessEstimation(estimation)

    def test_estimate_liveness_batch_with_threshold(self):
        """
        Test estimate liveness batch with threshold
        """
        qualityThreshold = 0.95
        detection = self.detector.detectOne(VLImage.load(filename=SPOOF))
        estimations = self.livenessEstimator.estimateBatch(
            [self.detection, detection], qualityThreshold=qualityThreshold
        )
        assert isinstance(estimations, list)
        assert len(estimations) == 2
        self.assertLivenessEstimation(estimations[0], LivenessPrediction.Real)
        self.assertLivenessEstimation(estimations[1], LivenessPrediction.Unknown)

    def test_estimate_liveness_batch_invalid_input(self):
        """
        Test estimate liveness batch with invalid input
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.livenessEstimator.estimateBatch([])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidSpanSize.format("Invalid span size"))
