from lunavl.sdk.estimators.face_estimators.warp_quality import Quality
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.detect_test_class import DetectTestClass
from tests.resources import ONE_FACE
from tests.schemas import jsonValidator, QUALITY_SCHEMA

QUALITY_PROPERTIES = [key for key in Quality.__dict__.keys() if not (key.startswith("_") or key is "asDict")]


class TestEstimateQuality(DetectTestClass):
    """
    Test estimate emotions.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.defaultDetector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)
        cls.warper = cls.faceEngine.createWarper()
        cls.qualityEstimator = cls.faceEngine.createWarpQualityEstimator()
        cls.warps = [cls.warper.warp(detector.detectOne(VLImage.load(filename=ONE_FACE))) for detector in cls.detectors]

    def assertQuality(self, quality: Quality):
        """
        Function checks if an instance is Quality class

        Args:
            quality: quality estimation object
        """
        assert isinstance(quality, Quality), f"{quality.__class__} is not {Quality}"
        for propertyName in QUALITY_PROPERTIES:
            property = getattr(quality, propertyName)
            assert isinstance(property, float), f"{propertyName} is not float"
            assert 0 <= property < 1, f"{propertyName} is out of range [0,1]"

    @staticmethod
    def assertQualityReply(qualityDict: dict):
        """
        Validate quality reply
        Args:
            qualityDict: quality estimation result
        """
        assert (
            jsonValidator(schema=QUALITY_SCHEMA).validate(qualityDict) is None
        ), f"{qualityDict} does not match with schema {QUALITY_SCHEMA}"

    def test_estimate_quality_(self):
        """
        Test quality estimations
        """
        for idx, detector in enumerate(self.detectors):
            with self.subTest(detectorType=detector.detectorType):
                quality = self.qualityEstimator.estimate(self.warps[idx].warpedImage)
                self.assertQuality(quality)

    def test_estimate_quality_as_dict(self):
        """
        Test quality estimations as dict
        """
        for idx, detector in enumerate(self.detectors):
            with self.subTest(detectorType=detector.detectorType):
                qualityDict = self.qualityEstimator.estimate(self.warps[idx].warpedImage).asDict()
                self.assertQualityReply(qualityDict)
