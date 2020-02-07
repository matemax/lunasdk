import pytest

from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.detect_test_class import DetectTestClass
from tests.resources import ONE_FACE


class TestEstimateEmotions(DetectTestClass):
    """
    Test estimate emotions.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.defaultDetector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)
        cls.warper = cls.faceEngine.createWarper()
        cls.gazeEstimator = cls.faceEngine.createGazeEstimator()
        cls.faceDetection = cls.defaultDetector.detectOne(VLImage.load(filename=ONE_FACE))
        cls.warp = cls.warper.warp(faceDetection=cls.faceDetection)

    @staticmethod
    def validate_gaze_estimation(receivedDict: dict):
        """
        Validate gaze estimation reply
        """
        assert sorted(["pitch", "yaw"]) == sorted(receivedDict.keys())
        for gaze in ("pitch", "yaw"):
            assert isinstance(receivedDict[gaze], float)
            assert -180 <= receivedDict[gaze] <= 180

    def test_estimate_gaze_landmarks5(self):
        """
        Test gaze estimator with landmarks 5
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                landMarks5Transformation = self.warper.makeWarpTransformationWithLandmarks(self.faceDetection, "L5")
                gazeEstimation = self.gazeEstimator.estimate(landMarks5Transformation, self.warp).asDict()
                self.validate_gaze_estimation(gazeEstimation)

    def test_estimate_gaze_landmarks68(self):
        """
        Test gaze estimator with landmarks 68 (not supported by estimator)
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                faceDetection = detector.detectOne(VLImage.load(filename=ONE_FACE), detect68Landmarks=True)
                landMarks68Transformation = self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L68")
                with pytest.raises(TypeError):
                    self.gazeEstimator.estimate(landMarks68Transformation, self.warp)

    def test_estimate_gaze_landmarks68_without_landmarks68_detection(self):
        """
        Test gaze estimator with landmarks 68
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                faceDetection = detector.detectOne(VLImage.load(filename=ONE_FACE), detect68Landmarks=False)
                with pytest.raises(ValueError):
                    self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L68")

    def test_estimate_gaze_landmarks_wrong(self):
        """
        Test gaze estimator with wrong landmarks
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(ValueError):
                    self.warper.makeWarpTransformationWithLandmarks(self.faceDetection, "L10")

    def test_estimate_gaze_landmarks68_without_transformation(self):
        """
        Test gaze estimator without transformation
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                faceDetection = detector.detectOne(VLImage.load(filename=ONE_FACE), detect68Landmarks=False)
                with pytest.raises(LunaSDKException):
                    self.gazeEstimator.estimate(faceDetection.landmarks5, self.warp)
