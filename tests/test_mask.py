from lunavl.sdk.estimators.face_estimators.mask import Mask, MaskEstimator
from lunavl.sdk.image_utils.image import VLImage
from sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from tests.base import BaseTestClass
from tests.resources import WARP_CLEAN_FACE, FACE_WITH_MASK, OCCLUDED_FACE, MASK_NOT_IN_PLACE
from tests.schemas import jsonValidator, MASK_SCHEMA

MASK_PROPERTIES = [key for key in Mask.__dict__.keys() if not (key.startswith("_") or key == "asDict")]


class TestMask(BaseTestClass):
    """
    Test estimate mask.
    """

    # warp mask estimator
    maskEstimator: MaskEstimator

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.maskEstimator = cls.faceEngine.createMaskEstimator()

        cls.warpImageWithMask = FaceWarpedImage(VLImage.load(filename=FACE_WITH_MASK))
        cls.warpImageNoMask = FaceWarpedImage(VLImage.load(filename=WARP_CLEAN_FACE))
        cls.warpImageMaskNotInPlace = FaceWarpedImage(VLImage.load(filename=MASK_NOT_IN_PLACE))
        cls.warpImageOccludedFace = FaceWarpedImage(VLImage.load(filename=OCCLUDED_FACE))

    @staticmethod
    def assertMaskEstimation(mask: Mask):
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

    def assertMaskPropertyResult(self, maskObj: Mask, expectedEstimationResults: dict):
        """
        Function checks predominant property from result

        Args:
            maskObj: mask estimation object
            expectedEstimationResults: dictionary with probability scores
        """
        for propName in expectedEstimationResults:
            self.assertAlmostEqual(
                getattr(maskObj, propName),
                expectedEstimationResults[propName],
                delta=0.001,
                msg=f"property value '{propName}' is incorrect",
            )

    def test_estimate_mask_as_dict(self):
        """
        Test mask estimations as dict
        """
        maskDict = TestMask.maskEstimator.estimate(self.warpImageWithMask).asDict()
        assert (
            jsonValidator(schema=MASK_SCHEMA).validate(maskDict) is None
        ), f"{maskDict} does not match with schema {MASK_SCHEMA}"

    def test_estimate_with_mask(self):
        """
        Test mask estimations with mask exists on the face
        """
        expectedResult = {"maskInPlace": 0.977, "maskNotInPlace": 0.022, "noMask": 0.001, "occludedFace": 0.001}
        mask = TestMask.maskEstimator.estimate(self.warpImageWithMask)
        self.assertMaskEstimation(mask)
        self.assertMaskPropertyResult(mask, expectedResult)

    def test_estimate_without_mask_on_the_face(self):
        """
        Test mask estimations without mask on the face
        """
        expectedResult = {"maskInPlace": 0.007, "maskNotInPlace": 0.071, "noMask": 0.897, "occludedFace": 0.025}
        mask = TestMask.maskEstimator.estimate(self.warpImageNoMask)
        self.assertMaskEstimation(mask)
        self.assertMaskPropertyResult(mask, expectedResult)

    def test_estimate_mask_not_in_place(self):
        """
        Test mask estimations with mask exists on the face and is not worn properly
        """
        expectedResult = {"maskInPlace": 0.042, "maskNotInPlace": 0.386, "noMask": 0.003, "occludedFace": 0.567}
        mask = TestMask.maskEstimator.estimate(self.warpImageMaskNotInPlace)
        self.assertMaskEstimation(mask)
        self.assertMaskPropertyResult(mask, expectedResult)

    def test_estimate_mask_occluded_face(self):
        """
        Test mask estimations with face is occluded by other object
        """
        expectedResult = {"maskInPlace": 0.001, "maskNotInPlace": 0.141, "noMask": 0.326, "occludedFace": 0.531}
        mask = TestMask.maskEstimator.estimate(self.warpImageOccludedFace)
        self.assertMaskEstimation(mask)
        self.assertMaskPropertyResult(mask, expectedResult)
