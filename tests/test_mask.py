from collections import namedtuple

from lunavl.sdk.estimators.face_estimators.mask import Mask, MaskEstimator
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import CLEAN_ONE_FACE, FACE_WITH_MASK, OCCLUDED_FACE, MASK_NOT_IN_PLACE
from tests.schemas import jsonValidator, MASK_SCHEMA

MASK_PROPERTIES = [key for key in Mask.__dict__.keys() if not (key.startswith("_") or key == "asDict")]

MaskCase = namedtuple("MaskCase", ("propertyResult", "excludedProperties"))
TEST_DATA_FOR_MASK = {
    "maskInPlace": MaskCase(0.95, {"noMask": 0.01, "maskNotInPlace": 0.01, "occludedFace": 0.01}),
    "noMask": MaskCase(0.95, {"maskInPlace": 0.01, "maskNotInPlace": 0.02, "occludedFace": 0.01}),
    "maskNotInPlace": MaskCase(0.6, {"maskInPlace": 0.2, "noMask": 0.01, "occludedFace": 0.2}),
    "occludedFace": MaskCase(0.3, {"maskInPlace": 0.01, "noMask": 0.5, "maskNotInPlace": 0.25}),
}


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

        defaultDetector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)
        warper = cls.faceEngine.createFaceWarper()
        cls.warpImageWithMask = warper.warp(defaultDetector.detectOne(VLImage.load(filename=FACE_WITH_MASK)))
        cls.warpImageNoMask = warper.warp(defaultDetector.detectOne(VLImage.load(filename=CLEAN_ONE_FACE)))
        cls.warpImageMaskNotInPlace = warper.warp(defaultDetector.detectOne(VLImage.load(filename=MASK_NOT_IN_PLACE)))
        cls.warpImageOccludedFace = warper.warp(defaultDetector.detectOne(VLImage.load(filename=OCCLUDED_FACE)))

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

    @staticmethod
    def assertMaskPropertyResult(maskObj: Mask, expectedPredominantProperty: str):
        """
        Function checks predominant property from result

        Args:
            maskObj: mask estimation object
            expectedPredominantProperty: expected property of the mask object
        """
        lowerProbabilitySet = set(MASK_PROPERTIES) - {expectedPredominantProperty}
        actualPropertyResult = getattr(maskObj, expectedPredominantProperty)
        expectedPropertyResult = TEST_DATA_FOR_MASK[expectedPredominantProperty].propertyResult
        assert (
            actualPropertyResult > expectedPropertyResult
        ), f"Value of the Mask estimation '{actualPropertyResult}' is less than '{expectedPropertyResult}'"
        for propName in lowerProbabilitySet:
            assert (
                getattr(maskObj, propName)
                < TEST_DATA_FOR_MASK[expectedPredominantProperty].excludedProperties[propName]
            )

    def test_estimate_mask(self):
        """
        Test mask estimations
        """
        for warp in (self.warpImageWithMask, self.warpImageWithMask.warpedImage):
            with self.subTest(warp=type(warp).__name__):
                mask = TestMask.maskEstimator.estimate(warp)
                self.assertMaskEstimation(mask)

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
        mask = TestMask.maskEstimator.estimate(self.warpImageWithMask)
        self.assertMaskEstimation(mask)
        self.assertMaskPropertyResult(mask, "maskInPlace")

    def test_estimate_without_mask_on_the_face(self):
        """
        Test mask estimations without mask on the face
        """
        mask = TestMask.maskEstimator.estimate(self.warpImageNoMask)
        self.assertMaskEstimation(mask)
        self.assertMaskPropertyResult(mask, "noMask")

    def test_estimate_mask_not_in_place(self):
        """
        Test mask estimations with mask exists on the face and is not worn properly
        """
        mask = TestMask.maskEstimator.estimate(self.warpImageMaskNotInPlace)
        self.assertMaskEstimation(mask)
        self.assertMaskPropertyResult(mask, "maskNotInPlace")

    def test_estimate_mask_occluded_face(self):
        """
        Test mask estimations with face is occluded by other object
        """
        mask = TestMask.maskEstimator.estimate(self.warpImageOccludedFace)
        self.assertMaskEstimation(mask)
        self.assertMaskPropertyResult(mask, "occludedFace")
