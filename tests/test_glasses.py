from typing import Dict

from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.estimators.face_estimators.glasses import Glasses, GlassesEstimator
from lunavl.sdk.image_utils.image import VLImage

from tests.base import BaseTestClass
from tests.resources import WARP_CLEAN_FACE, WARP_FACE_WITH_EYEGLASSES, WARP_FACE_WITH_SUNGLASSES
from tests.schemas import jsonValidator, GLASSES_SCHEMA


class TestGlasses(BaseTestClass):
    """
    Test estimate glasses.
    """

    glassesEstimator: GlassesEstimator

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.glassesEstimator = cls.faceEngine.createGlassesEstimator()

        cls.warpNoGlasses = FaceWarpedImage(VLImage.load(filename=WARP_CLEAN_FACE))
        cls.warpEyeGlasses = FaceWarpedImage(VLImage.load(filename=WARP_FACE_WITH_EYEGLASSES))
        cls.warpSunGlasses = FaceWarpedImage(VLImage.load(filename=WARP_FACE_WITH_SUNGLASSES))

    def assertGlassesEstimation(self, glasses: Glasses, expectedEstimationResults: Dict[str, str]):
        """
        Function checks if the instance belongs to the Glasses class and compares the result with what is expected.

        Args:
            glasses: glasses estimation object
            expectedEstimationResults: dictionary with result
        """
        assert isinstance(glasses, Glasses), f"{glasses.__class__} is not {Glasses}"
        self.assertEqual(glasses.asDict(), expectedEstimationResults)

    def test_estimate_glasses_as_dict(self):
        """
        Test glasses estimations as dict
        """
        glassesDict = TestGlasses.glassesEstimator.estimate(self.warpNoGlasses).asDict()
        assert (
            jsonValidator(schema=GLASSES_SCHEMA).validate(glassesDict) is None
        ), f"{glassesDict} does not match with schema {GLASSES_SCHEMA}"

    def test_estimate_no_glasses(self):
        """
        Test glasses estimations without glasses on the face
        """
        expectedResult = {"glasses": "no_glasses"}
        glasses = TestGlasses.glassesEstimator.estimate(self.warpNoGlasses)
        self.assertGlassesEstimation(glasses, expectedResult)

    def test_estimate_eye_glasses(self):
        """
        Test glasses estimations with eyeglasses on the face
        """
        expectedResult = {"glasses": "eyeglasses"}
        glasses = TestGlasses.glassesEstimator.estimate(self.warpEyeGlasses)
        self.assertGlassesEstimation(glasses, expectedResult)

    def test_estimate_sun_glasses(self):
        """
        Test glasses estimations with sunglasses on the face
        """
        expectedResult = {"glasses": "sunglasses"}
        glasses = TestGlasses.glassesEstimator.estimate(self.warpSunGlasses)
        self.assertGlassesEstimation(glasses, expectedResult)

    def test_async_estimate_glasses(self):
        """
        Test async estimate glasses
        """
        task = self.glassesEstimator.estimate(self.warpSunGlasses, asyncEstimate=True)
        self.assertAsyncEstimation(task, Glasses)
        task = self.glassesEstimator.estimateBatch([self.warpSunGlasses] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, Glasses)
