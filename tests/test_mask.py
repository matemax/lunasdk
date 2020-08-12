from lunavl.sdk.estimators.face_estimators.mask import Mask
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ONE_FACE
from tests.schemas import jsonValidator, MASK_SCHEMA

MASK_PROPERTIES = [key for key in Mask.__dict__.keys() if not (key.startswith("_") or key == "asDict")]


class TestMask(BaseTestClass):
    """
    Test estimate mask.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.maskEstimator = cls.faceEngine.createMaskEstimator()
        cls.warp = cls.warper.warp(cls.detector.detectOne(VLImage.load(filename=ONE_FACE)))

    def assertMask(self, mask: Mask):
        """
        Function checks if an instance is Mask class

        Args:
            mask: mask estimation object
        """
        assert isinstance(mask, Mask), f"{mask.__class__} is not {Mask}"
        for propertyName in MASK_PROPERTIES:
            property = getattr(mask, propertyName)
            assert isinstance(property, float), f"{propertyName} is not float"
            assert 0 <= property < 1, f"{propertyName} is out of range [0,1]"

    @staticmethod
    def assertMaskReply(maskDict: dict):
        """
        Validate mask reply
        Args:
            maskDict: mask estimation result
        """
        assert (
            jsonValidator(schema=MASK_SCHEMA).validate(maskDict) is None
        ), f"{maskDict} does not match with schema {MASK_SCHEMA}"

    def test_estimate_mask(self):
        """
        Test mask estimations
        """
        mask = self.maskEstimator.estimate(self.warp.warpedImage)
        self.assertMask(mask)

    def test_estimate_mask_as_dict(self):
        """
        Test mask estimations as dict
        """
        maskDict = self.maskEstimator.estimate(self.warp.warpedImage).asDict()
        self.assertMaskReply(maskDict)
