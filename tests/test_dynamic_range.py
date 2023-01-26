import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.faceengine.setting_provider import DetectorType
from tests.base import BaseTestClass
from tests.resources import CROWD_7_PEOPLE, CROWD_9_PEOPLE


class TestDynamicRange(BaseTestClass):
    """
    Test estimate dynamic range.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.dynamicRangeEstimator = cls.faceEngine.createDynamicRangeEstimator()
        cls.testImage = VLImage.load(filename=CROWD_7_PEOPLE)
        cls.testImage2 = VLImage.load(filename=CROWD_9_PEOPLE)

    def test_dynamic_range_single_image(self):
        """
        Test dynamic range with single image
        """
        faceDetection = self.detector.detectOne(self.testImage)
        dynamicRange = self.dynamicRangeEstimator.estimate(faceDetection)
        assert 0 <= dynamicRange <= 1

    def test_dynamic_range_known_value(self):
        """
        Test dynamic range with known value of image
        """
        faceDetection = self.detector.detectOne(self.testImage2)
        dynamicRange = self.dynamicRangeEstimator.estimate(faceDetection)
        assert 0.61 <= dynamicRange <= 0.62

    def test_dynamic_range_batch(self):
        """
        Test dynamic range batch
        """
        faceDetections = self.detector.detect([self.testImage, self.testImage2])
        dynamicRanges = self.dynamicRangeEstimator.estimateBatch([faceDetections[0][0], faceDetections[1][0]])
        for dynamicRange in dynamicRanges:
            assert 0 <= dynamicRange <= 1

    def test_dynamic_range_invalid_input(self):
        """
        Test dynamic range batch with invalid input
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.dynamicRangeEstimator.estimateBatch([])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidSpanSize.format("Invalid span size"))

    def test_async_estimate_dynamic_range(self):
        """
        Test async estimate dynamic range
        """
        faceDetection = self.detector.detectOne(self.testImage)
        task = self.dynamicRangeEstimator.estimate(faceDetection, asyncEstimate=True)
        self.assertAsyncEstimation(task, float)

    def test_async_estimate_dynamic_range_batch(self):
        """
        Test async estimate dynamic range batch
        """
        faceDetections = self.detector.detect([self.testImage, self.testImage2])
        task = self.dynamicRangeEstimator.estimateBatch(
            [faceDetections[0][0], faceDetections[1][0]], asyncEstimate=True
        )
        self.assertAsyncBatchEstimation(task, float)
