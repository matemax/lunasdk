import pytest

from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.detect_test_class import DetectTestClass
from tests.resources import EMOTION_FACES

EMOTION_IMAGES = {emotion: VLImage.load(filename=imagePath) for emotion, imagePath in EMOTION_FACES.items()}


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
        cls.faceDetection = cls.defaultDetector.detectOne(EMOTION_IMAGES["fear"])
        cls.warp = cls.warper.warp(faceDetection=cls.faceDetection)

    @staticmethod
    def validate_emotion_dict(receivedDict: dict):
        """
        Validate emotion dict
        """
        assert sorted(["pitch", "yaw"]) == sorted(receivedDict.keys())
        for gaze in ("pitch", "yaw"):
            assert -180 <= receivedDict[gaze] <= 180

    def test_estimate_gaze_landmarks5(self):
        """
        Test gaze estimator with landmarks 5
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                landMarks5Transformation = self.warper.makeWarpTransformationWithLandmarks(self.faceDetection, "L5")
                gazeEstimation = self.gazeEstimator.estimate(landMarks5Transformation, self.warp).asDict()
                self.validate_emotion_dict(gazeEstimation)

    def test_estimate_gaze_landmarks68(self):
        """
        Test gaze estimator with landmarks 68
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                faceDetection = detector.detectOne(EMOTION_IMAGES["fear"], detect68Landmarks=True)
                landMarks68Transformation = self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L68")
                with pytest.raises(TypeError):
                    self.gazeEstimator.estimate(landMarks68Transformation, self.warp)

    def test_estimate_gaze_landmarks_wrong(self):
        """
        Test gaze estimator with wrong landmarks
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                with pytest.raises(ValueError):
                    self.warper.makeWarpTransformationWithLandmarks(self.faceDetection, "L10")
