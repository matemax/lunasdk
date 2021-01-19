from collections import namedtuple
from typing import Optional

import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.livenessv1 import LivenessPrediction, LivenessV1
from lunavl.sdk.faceengine.engine import VLFaceEngine
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
        cls.headPose = cls.headPoseEstimator.estimate(cls.detection.landmarks68)

    def assert_liveness_estimation(
        self, estimation: LivenessV1, expectedPrediction: Optional[LivenessPrediction] = None
    ):
        """
        Assert estimation liveness.

        Args:
            estimation: estimation
            expectedPrediction: expected prediction
        """
        assert isinstance(estimation, LivenessV1), f"wrong estimation type {type(estimation)}"
        assert isinstance(estimation.prediction, LivenessPrediction), (
            f"wrong estimation type " f"{type(estimation.prediction)}"
        )
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
        glassesDict = self.livenessEstimator.estimate(self.detection).asDict()
        assert (
            jsonValidator(schema=LIVENESSV1_SCHEMA).validate(glassesDict) is None
        ), f"{glassesDict} does not match with schema {LIVENESSV1_SCHEMA}"

    def test_liveness_estimation(self):
        """
        Test eye estimator with face with opened eyes
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
                headPose = self.headPoseEstimator.estimate(faceDetection.landmarks68)
                estimation = self.livenessEstimator.estimate(faceDetection=faceDetection, headPose=headPose)
                self.assert_liveness_estimation(estimation, expectedPrediction=case.prediction)

    def test_estimate_liveness_by_small_detection(self):
        """
        Test estimate liveness by small face detection. Todo: remove after FSDK-2811
        """
        faceDetection = self.detector.detectOne(VLImage.load(filename=SMALL_IMAGE))
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.livenessEstimator.estimate(faceDetection=faceDetection)
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidInput.format(details="Invalid input"))

    def test_estimate_liveness_with_head_pose_angles_info(self):
        """
        Test estimate liveness with head pose angles info
        """
        for angle in ("yaw", "roll", "pitch"):
            for sign in (-1, 1):
                with self.subTest(angle=angle, sign=sign):
                    estimation = self.livenessEstimator.estimate(faceDetection=self.detection, **{angle: sign * 30})
                    self.assert_liveness_estimation(estimation, LivenessPrediction.Unknown)
                    estimation = self.livenessEstimator.estimate(faceDetection=self.detection, **{angle: sign * 1})
                    self.assert_liveness_estimation(estimation, LivenessPrediction.Real)

    def test_estimate_liveness_with_filtration_by_handle(self):
        """
        Test estimate liveness with head pose info
        """
        Case = namedtuple("Case", ("prediction", "principalAxes"))
        cases = (
            Case(LivenessPrediction.Real, 60),
            Case(LivenessPrediction.Unknown, 1),
            Case(LivenessPrediction.Real, None),
        )
        for case in cases:
            with self.subTest(prediction=case.prediction, principalAxes=case.principalAxes):
                livenessEstimator = self.faceEngine.createLivenessV1Estimator(principalAxes=case.principalAxes)
                estimation = livenessEstimator.estimate(faceDetection=self.detection, headPose=self.headPose)
                self.assert_liveness_estimation(estimation, case.prediction)

    def test_estimate_liveness_by_detection_on_image_border(self):
        """
        Test estimate liveness by face detection on an image border. Todo: remove after FSDK-2811
        """
        faceEngine = VLFaceEngine()
        faceEngine.faceEngineProvider.livenessV1Estimator.borderDistance = 1000
        detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)
        faceDetection = detector.detectOne(VLImage.load(filename=ONE_FACE))
        livenessEstimator = faceEngine.createLivenessV1Estimator()
        with pytest.raises(LunaSDKException) as exceptionInfo:
            livenessEstimator.estimate(faceDetection=faceDetection)
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidInput.format(details="Invalid input"))

    def test_estimate_liveness_by_detection_without_landmarks5(self):
        """
        Test estimate liveness by face detection without landmarks5. Todo: remove after FSDK-2811
        """
        faceDetection = self.detector.detectOne(VLImage.load(filename=SMALL_IMAGE), detect5Landmarks=False)
        with pytest.raises(ValueError) as exceptionInfo:
            self.livenessEstimator.estimate(faceDetection=faceDetection)
        assert "Landmarks5 is required for liveness estimation" == str(exceptionInfo), str(exceptionInfo)
