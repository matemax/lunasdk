from collections import namedtuple
from typing import Dict

from lunavl.sdk.detectors.facedetector import FaceDetector
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.estimators.face_estimators.mask import Mask, MaskEstimator
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import (
    WARP_CLEAN_FACE,
    FACE_WITH_MASK,
    OCCLUDED_FACE,
    FULL_FACE_NO_MASK,
    FULL_FACE_WITH_MASK,
    FULL_OCCLUDED_FACE,
    LARGE_IMAGE,
)
from tests.schemas import jsonValidator, MASK_SCHEMA

MaskProperties = namedtuple("MaskProperties", ("missing", "medicalMask", "occluded"))
TestCase = namedtuple("TestCase", ("name", "inputImage", "isWarp", "expectedResult", "rect"))


class TestMask(BaseTestClass):
    """
    Test estimate mask.
    """

    # warp mask estimator
    maskEstimator: MaskEstimator
    defaultDetector: FaceDetector

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.maskEstimator = cls.faceEngine.createMaskEstimator()

        cls.warpImageMedicalMask = FaceWarpedImage(VLImage.load(filename=FACE_WITH_MASK))
        cls.warpImageMissing = FaceWarpedImage(VLImage.load(filename=WARP_CLEAN_FACE))
        cls.warpImageOccluded = FaceWarpedImage(VLImage.load(filename=OCCLUDED_FACE))

        cls.imageMedicalMask = VLImage.load(filename=FULL_FACE_WITH_MASK)
        cls.imageMissing = VLImage.load(filename=FULL_FACE_NO_MASK)
        cls.imageOccluded = VLImage.load(filename=FULL_OCCLUDED_FACE)

        cls.largeImage = VLImage.load(filename=LARGE_IMAGE)

        cls.defaultDetector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)

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
        cases = [
            TestCase("medical_mask_warp", self.warpImageMedicalMask, True, MaskProperties(0.0, 0.999, 0.0), None),
            TestCase("medical_mask_image", self.imageMedicalMask, False, MaskProperties(0.0, 0.999, 0.0), None),
        ]
        for case in cases:
            with self.subTest(name=case.name):
                if case.isWarp:
                    mask = TestMask.maskEstimator.estimate(case.inputImage)
                else:
                    faceDetection = self.defaultDetector.detectOne(case.inputImage)
                    mask = TestMask.maskEstimator.estimate(faceDetection)
                self.assertMaskEstimation(mask, case.expectedResult._asdict())

    def test_estimate_missing_mask(self):
        """
        Test mask estimations without mask on the face
        """
        cases = [
            TestCase("no_mask_warp", self.warpImageMissing, True, MaskProperties(0.998, 0.001, 0.000), None),
            TestCase("no_mask_image", self.imageMissing, False, MaskProperties(0.997, 0.0, 0.001), None),
        ]
        for case in cases:
            with self.subTest(name=case.name):
                if case.isWarp:
                    mask = TestMask.maskEstimator.estimate(case.inputImage)
                else:
                    faceDetection = self.defaultDetector.detectOne(case.inputImage)
                    mask = TestMask.maskEstimator.estimate(faceDetection)
                self.assertMaskEstimation(mask, case.expectedResult._asdict())

    def test_estimate_mask_occluded(self):
        """
        Test mask estimations with face is occluded by other object
        """
        cases = [
            TestCase("occluded_warp", self.warpImageOccluded, True, MaskProperties(0.259, 0.669, 0.071), None),
            TestCase("occluded_image", self.imageOccluded, False, MaskProperties(0.0, 0.0, 0.999), None),
        ]
        for case in cases:
            with self.subTest(name=case.name):
                if case.isWarp:
                    mask = TestMask.maskEstimator.estimate(case.inputImage)
                else:
                    faceDetection = self.defaultDetector.detectOne(case.inputImage)
                    mask = TestMask.maskEstimator.estimate(faceDetection)
                self.assertMaskEstimation(mask, case.expectedResult._asdict())

    def test_estimate_missing_mask_large_image(self):
        """
        Test mask estimations without mask on the face
        """
        case = TestCase("no_mask_image", self.largeImage, False, MaskProperties(0.000, 0.0, 0.999), None)

        faceDetection = self.defaultDetector.detectOne(case.inputImage)
        mask = TestMask.maskEstimator.estimate(faceDetection)
        self.assertMaskEstimation(mask, case.expectedResult._asdict())
