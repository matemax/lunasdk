from collections import namedtuple
from typing import Dict

import pytest

from lunavl.sdk.detectors.facedetector import FaceDetector
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.estimators.face_estimators.mask import (
    Mask,
    MaskEstimator,
    FaceOcclusionState,
    _FaceOcclusion,
    MaskState,
)
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import (
    FACE_WITH_MASK,
    FULL_FACE_NO_MASK,
    FULL_FACE_WITH_MASK,
    FULL_OCCLUDED_FACE,
    LARGE_IMAGE,
    OCCLUDED_FACE,
    WARP_CLEAN_FACE,
    MASK_CHIN,
    MASK_FULL,
    MASK_NOT_IN_PLACE,
)
from tests.schemas import MASK_SCHEMA, jsonValidator

MaskProperties = namedtuple("MaskProperties", ("missing", "medicalMask", "occluded"))
WarpNExpectedProperties = namedtuple("WarpNExpectedProperties", ("warp", "expectedProperties"))
TestCase = namedtuple("TestCase", ("name", "inputImage", "isWarp", "expectedResult", "rect"))


class TestMask(BaseTestClass):
    """
    Test estimate mask.
    """

    # warp mask estimator
    maskEstimator: MaskEstimator
    detector: FaceDetector

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.maskEstimator = cls.faceEngine.createMaskEstimator()

        cls.medicalMaskWarpNProperties = WarpNExpectedProperties(
            FaceWarpedImage(VLImage.load(filename=FACE_WITH_MASK)), MaskProperties(0.000, 0.884, 0.112)
        )
        cls.missingMaskWarpNProperties = WarpNExpectedProperties(
            FaceWarpedImage(VLImage.load(filename=WARP_CLEAN_FACE)), MaskProperties(0.821, 0.002, 0.178)
        )
        cls.occludedMaskWarpNProperties = WarpNExpectedProperties(
            FaceWarpedImage(VLImage.load(filename=OCCLUDED_FACE)), MaskProperties(0.409, 0.018, 0.572)
        )
        cls.imageMedicalMask = VLImage.load(filename=FULL_FACE_WITH_MASK)
        cls.warpImageMedicalMask = FaceWarpedImage(VLImage.load(filename=FACE_WITH_MASK))
        cls.imageMissing = VLImage.load(filename=FULL_FACE_NO_MASK)
        cls.warpImageMissing = FaceWarpedImage(VLImage.load(filename=WARP_CLEAN_FACE))
        cls.imageOccluded = VLImage.load(filename=FULL_OCCLUDED_FACE)
        cls.warpImageOccluded = FaceWarpedImage(VLImage.load(filename=OCCLUDED_FACE))

        cls.largeImage = VLImage.load(filename=LARGE_IMAGE)

        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)

    def assertMaskEstimation(self, mask: Mask, expectedEstimationResults: Dict[str, float]):
        """
        Function checks if the instance belongs to the Mask class and compares the result with what is expected.

        Args:
            mask: mask estimation object
            expectedEstimationResults: dictionary with probability scores
        """
        assert isinstance(mask, Mask), f"{mask.__class__} is not {Mask}"

        assert isinstance(mask.faceOcclusion, _FaceOcclusion)

        for propertyName in expectedEstimationResults:
            actualPropertyResult = getattr(mask, propertyName)
            assert isinstance(actualPropertyResult, float), f"{propertyName} is not float"
            assert 0 <= actualPropertyResult <= 1, f"{propertyName} is out of range [0,1]"
            assert expectedEstimationResults[propertyName] == pytest.approx(
                actualPropertyResult, abs=0.005
            ), f"property value '{propertyName}' is incorrect"

    def test_estimate_mask_as_dict(self):
        """
        Test mask estimations as dict
        """
        maskDict = TestMask.maskEstimator.estimate(self.medicalMaskWarpNProperties.warp).asDict()
        assert (
            jsonValidator(schema=MASK_SCHEMA).validate(maskDict) is None
        ), f"{maskDict} does not match with schema {MASK_SCHEMA}"

    def test_estimate_mask_correctness(self):
        """
        Test mask estimations without mask on the face
        """
        cases = [
            TestCase("no_mask_warp", self.warpImageMissing, True, MaskState.Missing, None),
            TestCase("no_mask_image", self.imageMissing, False, MaskState.Missing, None),
            TestCase("occluded_warp", self.warpImageOccluded, True, MaskState.Occluded, None),
            TestCase("occluded_image", self.imageOccluded, False, MaskState.Occluded, None),
            TestCase("medical_mask_warp", self.warpImageMedicalMask, True, MaskState.MedicalMask, None),
            TestCase("medical_mask_image", self.imageMedicalMask, False, MaskState.MedicalMask, None),
        ]
        for case in cases:
            with self.subTest(name=case.name):
                if case.isWarp:
                    mask = TestMask.maskEstimator.estimate(case.inputImage)
                else:
                    faceDetection = self.detector.detectOne(case.inputImage)
                    mask = TestMask.maskEstimator.estimate(faceDetection)
                assert case.expectedResult == mask.predominateMask

    def test_estimate_mask_batch(self):
        """
        Test batch mask estimation
        """
        warps = [self.medicalMaskWarpNProperties, self.missingMaskWarpNProperties, self.occludedMaskWarpNProperties]
        masks = self.maskEstimator.estimateBatch([warp.warp for warp in warps])
        assert isinstance(masks, list)
        assert len(masks) == len(warps)
        for idx, mask in enumerate(masks):
            self.assertMaskEstimation(mask, warps[idx].expectedProperties._asdict())

    def test_estimate_mask_batch_invalid_input(self):
        """
        Test batch mask estimation with invalid input
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.maskEstimator.estimateBatch([])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidSpanSize.format("Invalid span size"))

    def test_estimate_mask_on_large_image(self):
        """
        Test mask estimations on large image
        """
        faceDetection = self.detector.detectOne(self.largeImage)
        mask = TestMask.maskEstimator.estimate(faceDetection)
        assert MaskState.Occluded == mask.predominateMask

    def test_estimate_face_occlusion_by_mask(self):
        """
        Test occlusion face by mask
        """

        TestCase = namedtuple("TestCase", ("occlusion", "image"))
        cases = (
            TestCase(FaceOcclusionState.Chin, MASK_CHIN),
            TestCase(FaceOcclusionState.Mouth, MASK_NOT_IN_PLACE),
            TestCase(FaceOcclusionState.Correct, FACE_WITH_MASK),
            TestCase(FaceOcclusionState.Partially, OCCLUDED_FACE),
            TestCase(FaceOcclusionState.Clear, WARP_CLEAN_FACE),
            TestCase(FaceOcclusionState.Full, MASK_FULL),
        )
        warper = self.faceEngine.createFaceWarper()

        for case in cases:
            with self.subTest(case.occlusion):
                detection = self.detector.detectOne(VLImage.load(filename=case.image))
                warp = warper.warp(detection)
                mask = self.maskEstimator.estimate(warp)
                assert case.occlusion == mask.faceOcclusion.predominantOcclusion
                # check that we use correct DetailedMaskType in score properties
                assert getattr(mask.faceOcclusion, case.occlusion.name.lower()) > 0.4

    def test_async_estimate_mask(self):
        """
        Test async estimate mask
        """
        task = self.maskEstimator.estimate(self.warpImageMedicalMask, asyncEstimate=True)
        self.assertAsyncEstimation(task, Mask)
        task = self.maskEstimator.estimateBatch([self.warpImageMedicalMask] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, Mask)
