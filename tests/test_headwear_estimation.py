import snakecase

from lunavl.sdk.estimators.face_estimators.headwear import HeadwearType, Headwear
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import HAT, HOOD, BASEBALL_CAP, USHANKA, SHAWL, BEANIE, HELMET, ONE_FACE, PEAKED_CAP


class TestHeadwear(BaseTestClass):
    """
    Test estimate headwear.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.headwearEstimator = cls.faceEngine.createHeadwearEstimator()

    def test_estimate_headwear_correctness(self):
        """
        Test headwear estimator correctness
        """
        images = {
            HeadwearType.Hat: HAT,
            HeadwearType.Hood: HOOD,
            HeadwearType.BaseballCap: BASEBALL_CAP,
            HeadwearType.HatWithEarFlaps: USHANKA,
            HeadwearType.Shawl: SHAWL,
            HeadwearType.Beanie: BEANIE,
            HeadwearType.PeakedCap: PEAKED_CAP,
            HeadwearType.Helmet: HELMET,
            HeadwearType.NoHeadWear: ONE_FACE,
        }
        for headwearType, image in images.items():
            with self.subTest(headwearType):
                estimation = self.estimate(image)
                assert headwearType == estimation.type
                if headwearType != HeadwearType.NoHeadWear:
                    assert estimation.asDict()["type"] == snakecase.convert(headwearType.name)
                else:
                    assert estimation.asDict()["type"] == "none"

    def estimate(self, image: str = HOOD) -> Headwear:
        """Estimate headwear on image"""
        faceDetection = self.detector.detectOne(VLImage.load(filename=image))
        warp = self.warper.warp(faceDetection)
        estimation = self.headwearEstimator.estimate(warp)
        assert isinstance(estimation, Headwear)
        return estimation

    def test_estimate_headwear(self):
        """
        Simple headwear estimation
        """
        estimation = self.estimate(HAT)
        assert estimation.type == HeadwearType.Hat

    def test_headwear_as_dict(self):
        """
        Test method Headwear.asDict
        """
        estimation = self.estimate(HAT)
        assert {"type": "hat", "is_wear": True} == estimation.asDict()

    def test_estimate_headwear_batch(self):
        """
        Batch headwear estimation test
        """
        faceDetections = self.detector.detect([VLImage.load(filename=HAT), VLImage.load(filename=HOOD)])
        warp1 = self.warper.warp(faceDetections[0][0])
        warp2 = self.warper.warp(faceDetections[1][0])
        estimations = self.headwearEstimator.estimateBatch([warp1, warp2])
        assert HeadwearType.Hat == estimations[0].type
        assert HeadwearType.Hood == estimations[1].type
