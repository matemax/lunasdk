from lunavl.sdk.estimators.face_estimators.natural_light import FaceNaturalLight
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import BLACK_AND_WHITE, ONE_FACE


class TestFaceNaturalLight(BaseTestClass):
    """
    Test estimate image color type.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.faceNaturalLightEstimator = cls.faceEngine.createFaceNaturalLightEstimator()

    def test_estimate_face_natural_light_correctness(self):
        """
        Test image color type estimator correctness
        """
        estimation = self.estimate(BLACK_AND_WHITE)
        assert not estimation.status
        assert 0 <= estimation.score <= 1

    def estimate(self, image: str = BLACK_AND_WHITE) -> FaceNaturalLight:
        """Estimate face natural light on image"""
        faceDetection = self.detector.detectOne(VLImage.load(filename=image))
        warp = self.warper.warp(faceDetection)
        estimation = self.faceNaturalLightEstimator.estimate(warp)
        assert isinstance(estimation, FaceNaturalLight)
        return estimation

    def test_estimate_face_natural_light(self):
        """
        Simple image color type estimation
        """
        estimation = self.estimate(ONE_FACE)
        assert estimation.status
        assert 0 <= estimation.score <= 1

    def test_face_natural_light_as_dict(self):
        """
        Test method FaceNaturalLight.asDict
        """
        estimation = self.estimate(ONE_FACE)
        assert {
            "status": estimation.status,
            "score": estimation.score,
        } == estimation.asDict()

    def test_estimate_face_natural_light_batch(self):
        """
        Batch face natural light estimation test
        """
        faceDetections = self.detector.detect([VLImage.load(filename=ONE_FACE), VLImage.load(filename=BLACK_AND_WHITE)])
        warp1 = self.warper.warp(faceDetections[0][0])
        warp2 = self.warper.warp(faceDetections[1][0])
        estimations = self.faceNaturalLightEstimator.estimateBatch([warp1, warp2])
        assert estimations[0].status
        assert not estimations[1].status
