from collections import namedtuple
from typing import Dict

from lunavl.sdk.estimators.face_estimators.mask import Mask, MaskEstimator
from lunavl.sdk.image_utils.image import VLImage
from sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from tests.base import BaseTestClass
from tests.resources import WARP_CLEAN_FACE, FACE_WITH_MASK, OCCLUDED_FACE, MASK_NOT_IN_PLACE
from tests.schemas import jsonValidator, MASK_SCHEMA

MaskProperties = namedtuple("MaskProperties", ("maskInPlace", "maskNotInPlace", "noMask", "occludedFace"))


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

    def assertMaskEstimation(self, mask: Mask, expectedEstimationResults: Dict[str, float]):
        """
        Function checks if the instance belongs to the Mask class and compares the result with what is expected.

        Args:
            mask: mask estimation object
            expectedEstimationResults: dictionary with probability scores
        """
        assert isinstance(mask, Mask), f"{mask.__class__} is not {Mask}"

        for propertyName in expectedEstimationResults:
            with self.subTest(propertyName=propertyName):
                actualPropertyResult = getattr(mask, propertyName)
                assert isinstance(actualPropertyResult, float), f"{propertyName} is not float"
                assert 0 <= actualPropertyResult < 1, f"{propertyName} is out of range [0,1]"
                self.assertAlmostEqual(
                    actualPropertyResult,
                    expectedEstimationResults[propertyName],
                    delta=0.001,
                    msg=f"property value '{propertyName}' is incorrect",
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
        expectedResult = MaskProperties(0.977, 0.022, 0.001, 0.001)
        mask = TestMask.maskEstimator.estimate(self.warpImageWithMask)
        self.assertMaskEstimation(mask, expectedResult._asdict())

    def test_estimate_without_mask_on_the_face(self):
        """
        Test mask estimations without mask on the face
        """
        expectedResult = MaskProperties(0.007, 0.071, 0.897, 0.025)
        mask = TestMask.maskEstimator.estimate(self.warpImageNoMask)
        self.assertMaskEstimation(mask, expectedResult._asdict())

    def test_estimate_mask_not_in_place(self):
        """
        Test mask estimations with mask exists on the face and is not worn properly
        """
        expectedResult = MaskProperties(0.042, 0.386, 0.003, 0.567)
        mask = TestMask.maskEstimator.estimate(self.warpImageMaskNotInPlace)
        self.assertMaskEstimation(mask, expectedResult._asdict())

    def test_estimate_mask_occluded_face(self):
        """
        Test mask estimations with face is occluded by other object
        """
        expectedResult = MaskProperties(0.001, 0.141, 0.326, 0.531)
        mask = TestMask.maskEstimator.estimate(self.warpImageOccludedFace)
        self.assertMaskEstimation(mask, expectedResult._asdict())
