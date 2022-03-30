from lunavl.sdk.estimators.face_estimators.image_type import ImageColorType, ImageColorSchema
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import BLACK_AND_WHITE, ONE_FACE


class TestImageColorType(BaseTestClass):
    """
    Test estimate image color type.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.imageColorTypeEstimator = cls.faceEngine.createImageColorTypeEstimator()

    def test_estimate_image_color_type_correctness(self):
        """
        Test image color type estimator correctness
        """
        estimation = self.estimate(BLACK_AND_WHITE)
        assert ImageColorSchema.Grayscale == estimation.type
        assert 0 <= estimation.infrared <= 1
        assert 0 <= estimation.grayscale <= 1

    def estimate(self, image: str = BLACK_AND_WHITE) -> ImageColorType:
        """Estimate image color type on image"""
        faceDetection = self.detector.detectOne(VLImage.load(filename=image))
        warp = self.warper.warp(faceDetection)
        estimation = self.imageColorTypeEstimator.estimate(warp)
        assert isinstance(estimation, ImageColorType)
        return estimation

    def test_estimate_image_color_type(self):
        """
        Simple image color type estimation
        """
        estimation = self.estimate(ONE_FACE)
        assert estimation.type == ImageColorSchema.Color

    def test_image_color_type_as_dict(self):
        """
        Test method ImageColorType.asDict
        """
        estimation = self.estimate(ONE_FACE)
        assert {
            "grayscale": estimation.grayscale,
            "infrared": estimation.infrared,
            "type": "color",
        } == estimation.asDict()
        estimation = self.estimate(BLACK_AND_WHITE)
        assert {
            "grayscale": estimation.grayscale,
            "infrared": estimation.infrared,
            "type": "grayscale",
        } == estimation.asDict()

    def test_estimate_image_color_type_batch(self):
        """
        Batch image color type estimation test
        """
        faceDetections = self.detector.detect([VLImage.load(filename=ONE_FACE), VLImage.load(filename=BLACK_AND_WHITE)])
        warp1 = self.warper.warp(faceDetections[0][0])
        warp2 = self.warper.warp(faceDetections[1][0])
        estimations = self.imageColorTypeEstimator.estimateBatch([warp1, warp2])
        assert ImageColorSchema.Color == estimations[0].type
        assert ImageColorSchema.Grayscale == estimations[1].type
