from lunavl.sdk.estimators.face_estimators.fisheye import Fisheye
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import FISHEYE, FROWNING


class TestFisheyeEffect(BaseTestClass):
    """
    Test fisheye estimation.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.fisheyeEstimator = cls.faceEngine.createFisheyeEstimator()

    def test_estimate_fisheye_correctness(self):
        """
        Test fisheye estimator correctness
        """
        estimation = self.estimate(FISHEYE)
        assert estimation.status
        assert 0 <= estimation.score <= 1

    def estimate(self, image: str = FROWNING) -> Fisheye:
        """Estimate fisheye on image"""
        faceDetection = self.detector.detectOne(VLImage.load(filename=image))
        estimation = self.fisheyeEstimator.estimate(faceDetection)
        assert isinstance(estimation, Fisheye)
        return estimation

    def test_estimate_fisheye(self):
        """
        Simple fisheye estimation
        """
        estimation = self.estimate(FROWNING)
        assert not estimation.status
        assert 0 <= estimation.score <= 1

    def test_fisheye_as_dict(self):
        """
        Test method Fisheye.asDict
        """
        estimation = self.estimate(FROWNING)
        assert {
            "status": estimation.status,
            "score": estimation.score,
        } == estimation.asDict()

    def test_estimate_fisheye_batch(self):
        """
        Batch fisheye estimation test
        """
        faceDetections = self.detector.detect([VLImage.load(filename=FROWNING), VLImage.load(filename=FISHEYE)])
        estimations = self.fisheyeEstimator.estimateBatch([faceDetections[0][0], faceDetections[1][0]])
        assert not estimations[0].status
        assert estimations[1].status
