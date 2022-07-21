import pytest

from lunavl.sdk.base import BoundingBox
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.base import ImageWithFaceDetection
from lunavl.sdk.estimators.face_estimators.background import FaceDetectionBackground
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import MASK_CHIN, ONE_FACE

from FaceEngine import Detection, RectFloat


class TestFaceDetectionBackgroundEffect(BaseTestClass):
    """
    Test face detection background estimation.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.backgroundEstimator = cls.faceEngine.createFaceDetectionBackgroundEstimator()

    @staticmethod
    def assertEstimation(estimation: FaceDetectionBackground):
        assert isinstance(estimation, FaceDetectionBackground)
        assert 0 <= estimation.solidColor <= 1
        assert 0 <= estimation.lightBackground <= 1

    def test_estimate_background_correctness(self):
        """
        Test background estimator correctness
        """
        estimation = self.estimate(MASK_CHIN)
        assert estimation.status
        estimation = self.estimate(ONE_FACE)
        assert not estimation.status

    def estimate(self, image: str = ONE_FACE) -> FaceDetectionBackground:
        """Estimate fisheye on image"""
        faceDetection = self.detector.detectOne(VLImage.load(filename=image))
        estimation = self.backgroundEstimator.estimate(faceDetection)
        self.assertEstimation(estimation)
        return estimation

    def test_estimate_background(self):
        """
        Simple face detection background estimation
        """
        estimation = self.estimate(ONE_FACE)
        assert not estimation.status

    def test_background_as_dict(self):
        """
        Test method FaceDetectionBackground.asDict
        """
        estimation = self.estimate(ONE_FACE)
        assert {
            "light_background": estimation.lightBackground,
            "status": estimation.status,
            "solid_color": estimation.solidColor,
        } == estimation.asDict()

    def test_estimate_background_batch(self):
        """
        Batch face detection background estimation test
        """
        faceDetections = self.detector.detect([VLImage.load(filename=ONE_FACE), VLImage.load(filename=MASK_CHIN)])
        estimations = self.backgroundEstimator.estimateBatch([faceDetections[0][0], faceDetections[1][0]])
        for estimation in estimations:
            self.assertEstimation(estimation)
        assert not estimations[0].status
        assert estimations[1].status

    def test_async_estimate_background(self):
        """
        Test async estimate background
        """
        faceDetection = self.detector.detectOne(VLImage.load(filename=MASK_CHIN))

        task = self.backgroundEstimator.estimate(faceDetection, asyncEstimate=True)
        self.assertAsyncEstimation(task, FaceDetectionBackground)
        task = self.backgroundEstimator.estimateBatch([faceDetection] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, FaceDetectionBackground)

    def test_estimate_background_batch_invalid_input(self):
        """
        Test batch background estimator with invalid input
        """
        with pytest.raises(LunaSDKException) as e:
            self.backgroundEstimator.estimateBatch([], [])
        assert e.value.error.errorCode == LunaVLError.InvalidSpanSize.errorCode

    def test_estimate_background_by_image_and_bounding_box_without_intersection(self):
        """
        Estimating background by image and bounding box without intersection
        """
        fakeDetection = Detection(RectFloat(3000.0, 3000.0, 100.0, 100.0), 0.9)
        bBox = BoundingBox(fakeDetection)
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.backgroundEstimator.estimate(ImageWithFaceDetection(VLImage.load(filename=ONE_FACE), bBox))
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidRect.format("Invalid rectangle"))

    def test_estimate_background_by_image_and_bounding_box_empty_bounding_box(self):
        """
        Estimating background by image and empty bounding box
        """
        fakeDetection = Detection(RectFloat(0.0, 0.0, 0.0, 0.0), 0.9)
        bBox = BoundingBox(fakeDetection)
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.backgroundEstimator.estimate(ImageWithFaceDetection(VLImage.load(filename=ONE_FACE), bBox))
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidDetection.format("Invalid detection"))
