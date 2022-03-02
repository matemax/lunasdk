import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ONE_FACE
from lunavl.sdk.estimators.face_estimators.eyes import WarpWithLandmarks5


class TestEstimateGazeDirection(BaseTestClass):
    """
    Test estimate gaze direction.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.gazeEstimator = cls.faceEngine.createGazeEstimator()
        cls.faceDetection = cls.detector.detectOne(VLImage.load(filename=ONE_FACE))
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

        landMarks5Transformation = self.warper.makeWarpTransformationWithLandmarks(self.faceDetection, "L5")
        warpWithLandmarks5 = WarpWithLandmarks5(self.warp, landMarks5Transformation)
        gazeEstimation = self.gazeEstimator.estimate(warpWithLandmarks5).asDict()
        self.validate_gaze_estimation(gazeEstimation)

    def test_estimate_gaze_landmarks68(self):
        """
        Test gaze estimator with landmarks 68 (not supported by estimator)
        """

        faceDetection = self.detector.detectOne(VLImage.load(filename=ONE_FACE), detect68Landmarks=True)
        landMarks68Transformation = self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L68")
        warpWithLandmarks5 = WarpWithLandmarks5(self.warp, landMarks68Transformation)
        with pytest.raises(TypeError):
            self.gazeEstimator.estimate(warpWithLandmarks5)

    def test_estimate_gaze_landmarks68_without_landmarks68_detection(self):
        """
        Test gaze estimator with landmarks 68
        """
        faceDetection = self.detector.detectOne(VLImage.load(filename=ONE_FACE), detect68Landmarks=False)
        with pytest.raises(ValueError):
            self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L68")

    def test_estimate_gaze_landmarks_wrong(self):
        """
        Test gaze estimator with wrong landmarks
        """
        with pytest.raises(ValueError):
            self.warper.makeWarpTransformationWithLandmarks(self.faceDetection, "L10")

    def test_estimate_gaze_without_transformation(self):
        """
        Test gaze estimator without transformation
        """
        faceDetection = self.detector.detectOne(VLImage.load(filename=ONE_FACE), detect68Landmarks=False)
        warpWithLandmarks5 = WarpWithLandmarks5(self.warp, faceDetection.landmarks5)
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.gazeEstimator.estimate(warpWithLandmarks5)
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidLandmarks5.format("Invalid landmarks 5"))

    def test_batch_estimate_gaze(self):
        """
        Test gaze estimator with two warps
        """
        faceDetection = self.detector.detectOne(VLImage.load(filename=ONE_FACE))
        warp = self.warper.warp(faceDetection=faceDetection)
        warpWithLandmarks5List = [
            WarpWithLandmarks5(self.warp, self.warper.makeWarpTransformationWithLandmarks(self.faceDetection, "L5")),
            WarpWithLandmarks5(warp, self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")),
        ]
        gazeEstimations = self.gazeEstimator.estimateBatch(warpWithLandmarks5List)
        assert isinstance(gazeEstimations, list)
        assert len(gazeEstimations) == 2
        for gazeEstimation in gazeEstimations:
            self.validate_gaze_estimation(gazeEstimation.asDict())

    def test_batch_estimate_invalid_input(self):
        """
        Test gaze estimator with invalid input
        """
        with pytest.raises(LunaSDKException) as e:
            self.gazeEstimator.estimateBatch([], [])
        assert e.value.error.errorCode == LunaVLError.InvalidSpanSize.errorCode
