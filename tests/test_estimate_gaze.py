import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ONE_FACE


class TestEstimateGazeDirection(BaseTestClass):
    """
    Test estimate gaze direction.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)
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
        gazeEstimation = self.gazeEstimator.estimate(landMarks5Transformation, self.warp).asDict()
        self.validate_gaze_estimation(gazeEstimation)

    def test_estimate_gaze_landmarks68(self):
        """
        Test gaze estimator with landmarks 68 (not supported by estimator)
        """

        faceDetection = self.detector.detectOne(VLImage.load(filename=ONE_FACE), detect68Landmarks=True)
        landMarks68Transformation = self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L68")
        with pytest.raises(TypeError):
            self.gazeEstimator.estimate(landMarks68Transformation, self.warp)

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
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.gazeEstimator.estimate(faceDetection.landmarks5, self.warp)
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidInput)

    def test_batch_estimate_gaze(self):
        """
        Test gaze estimator with two warps
        """
        faceDetection = self.detector.detectOne(VLImage.load(filename=ONE_FACE))
        warp = self.warper.warp(faceDetection=faceDetection)
        landMarksTransformationList = [
            self.warper.makeWarpTransformationWithLandmarks(detection, "L5")
            for detection in (self.faceDetection, faceDetection)
        ]
        gazeEstimations = self.gazeEstimator.estimateBatch(landMarksTransformationList, [self.warp, warp])
        assert isinstance(gazeEstimations, list)
        assert len(gazeEstimations) == 2
        for gazeEstimation in gazeEstimations:
            self.validate_gaze_estimation(gazeEstimation.asDict())

    def test_batch_estimate_gaze_with_error(self):
        """
        Test gaze estimator with two warps and one landmarks transformation
        """
        faceDetection = self.detector.detectOne(VLImage.load(filename=ONE_FACE), detect68Landmarks=True)
        warp = self.warper.warp(faceDetection=faceDetection)
        landMarksTransformationList = [self.warper.makeWarpTransformationWithLandmarks(self.faceDetection, "L5")]
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.gazeEstimator.estimateBatch(landMarksTransformationList, [self.warp, warp])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidInput)
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.Ok.format("Ok"))
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[1], LunaVLError.Ok.format("Ok"))

    def test_batch_estimate_invalid_input(self):
        """
        Test gaze estimator with invalid input
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.gazeEstimator.estimateBatch([], [])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidInput)
