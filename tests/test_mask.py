from collections import namedtuple
from typing import Dict

from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.estimators.face_estimators.mask import Mask, MaskEstimator
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import WARP_CLEAN_FACE, FACE_WITH_MASK, OCCLUDED_FACE
from tests.schemas import jsonValidator, MASK_SCHEMA

MaskProperties = namedtuple("MaskProperties", ("missing", "medicalMask", "occluded"))


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

        cls.warpImageMedicalMask = FaceWarpedImage(VLImage.load(filename=FACE_WITH_MASK))
        cls.warpImageMissing = FaceWarpedImage(VLImage.load(filename=WARP_CLEAN_FACE))
        cls.warpImageOccluded = FaceWarpedImage(VLImage.load(filename=OCCLUDED_FACE))

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
        maskDict = TestMask.maskEstimator.estimate(self.warpImageMedicalMask).asDict()
        assert (
            jsonValidator(schema=MASK_SCHEMA).validate(maskDict) is None
        ), f"{maskDict} does not match with schema {MASK_SCHEMA}"

    def test_estimate_medical_mask(self):
        """
        Test mask estimations with mask exists on the face
        """
        expectedResult = MaskProperties(0.0, 0.999, 0.000)
        mask = TestMask.maskEstimator.estimate(self.warpImageMedicalMask)
        self.assertMaskEstimation(mask, expectedResult._asdict())

    def test_estimate_missing_mask(self):
        """
        Test mask estimations without mask on the face
        """
        expectedResult = MaskProperties(0.896, 0.078, 0.024)
        mask = TestMask.maskEstimator.estimate(self.warpImageMissing)
        self.assertMaskEstimation(mask, expectedResult._asdict())

        self.assertMaskEstimation(mask, expectedResult._asdict())

    def test_estimate_mask_occluded(self):
        """
        Test mask estimations with face is occluded by other object
        """
        expectedResult = MaskProperties(0.326, 0.142, 0.531)
        mask = TestMask.maskEstimator.estimate(self.warpImageOccluded)
        self.assertMaskEstimation(mask, expectedResult._asdict())
