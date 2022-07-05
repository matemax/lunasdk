from typing import List

from lunavl.sdk.estimators.body_estimators.body_attributes import (
    BodyAttributes,
    OutwearColorEnum,
    SleeveLength,
    HeadwearStateEnum,
    ApparentGenderEnum,
    Sleeve,
    HeadwearState,
    OutwearColor,
    ApparentGender,
    BackpackState,
    BackpackStateEnum,
)
from lunavl.sdk.estimators.body_estimators.humanwarper import HumanWarpedImage
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import (
    ONE_FACE,
    CLEAN_ONE_FACE,
    SEVERAL_FACES,
    T_SHORT,
    LONG_SLEEVE,
    RED_EYES,
    BACKPACK,
    TURNED_HEAD_POSE_FACE,
    HOOD,
    PALETTE_MODE,
    FROWNING,
    BAD_THRESHOLD_WARP,
    FULL_OCCLUDED_FACE,
    SHAWL,
    RAISED,
    ANGER,
    YELLOW,
    PINK,
    BLACK,
    RED,
    HUMAN_WARP,
)


class TestBodyAttributes(BaseTestClass):
    """
    Test body attributes estimation.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createHumanDetector()
        cls.warper = cls.faceEngine.createHumanWarper()
        cls.bodyAttributesEstimator = cls.faceEngine.createBodyAttributesEstimator()

    def test_estimate_body_attributes(self):
        """
        Test simple body attributes estimato
        """
        estimation = self.bodyAttributesEstimator.estimate(HumanWarpedImage.load(filename=HUMAN_WARP))
        assert isinstance(estimation, BodyAttributes)

        assert isinstance(estimation.outwearColor, OutwearColor)
        assert set(estimation.outwearColor.colors) == {
            OutwearColorEnum.White,
            OutwearColorEnum.Orange,
            OutwearColorEnum.Red,
        }
        assert isinstance(estimation.sleeve, Sleeve)
        assert estimation.sleeve.predominantState == SleeveLength.Unknown

        assert isinstance(estimation.headwear, HeadwearState)
        assert estimation.headwear.predominantState == HeadwearStateEnum.Unknown

        assert isinstance(estimation.apparentGender, ApparentGender)
        assert estimation.apparentGender.predominantGender == ApparentGenderEnum.Female

        assert isinstance(estimation.apparentAge, float)
        assert round(estimation.apparentAge) == 14

        assert isinstance(estimation.backpack, BackpackState)
        assert estimation.backpack.predominantState == BackpackStateEnum.Unknown

    def estimate(self, image: str = ONE_FACE) -> List[BodyAttributes]:
        """Estimate body attributes on image"""
        detections = self.detector.detect([VLImage.load(filename=image)])[0]
        warps = [self.warper.warp(bodyDetection) for bodyDetection in detections]
        estimations = self.bodyAttributesEstimator.estimateBatch(warps)
        for estimation in estimations:
            assert isinstance(estimation, BodyAttributes)
        return estimations

    def test_face_body_attributes_as_dict(self):
        """
        Test method BodyAttributes.asDict
        """
        estimation = self.estimate(ONE_FACE)[0]
        assert {
            "apparent_age": round(estimation.apparentAge),
            "apparent_gender": {
                "predominant_gender": str(estimation.apparentGender.predominantGender.name).lower(),
                "estimations": {
                    "male": estimation.apparentGender.male,
                    "female": estimation.apparentGender.female,
                    "unknown": estimation.apparentGender.unknown,
                },
            },
            "backpack": {
                "predominant_state": str(estimation.backpack.predominantState.name).lower(),
                "estimations": {
                    "yes": estimation.backpack.yes,
                    "no": estimation.backpack.no,
                    "unknown": estimation.backpack.unknown,
                },
            },
            "headwear": {
                "predominant_state": str(estimation.headwear.predominantState.name).lower(),
                "estimations": {
                    "yes": estimation.headwear.yes,
                    "no": estimation.headwear.no,
                    "unknown": estimation.headwear.unknown,
                },
            },
            "sleeve": {
                "predominant_state": str(estimation.sleeve.predominantState.name).lower(),
                "estimations": {
                    "long": estimation.sleeve.long,
                    "short": estimation.sleeve.short,
                    "unknown": estimation.sleeve.unknown,
                },
            },
            "outwear_color": [color.value for color in estimation.outwearColor.colors],
        } == estimation.asDict()

    def test_estimate_body_attributes_batch(self):
        """
        Batch body attributes estimation test
        """
        bodyDetections = self.detector.detect([VLImage.load(filename=ONE_FACE), VLImage.load(filename=CLEAN_ONE_FACE)])
        warp1 = self.warper.warp(bodyDetections[0][0])
        warp2 = self.warper.warp(bodyDetections[1][0])
        estimations = self.bodyAttributesEstimator.estimateBatch([warp1, warp2])
        assert estimations[0].outwearColor.colors != estimations[1].outwearColor.colors

    def test_async_estimate_body_attributes(self):
        """
        Test async estimate body attributes
        """
        bodyDetections = self.detector.detect([VLImage.load(filename=ONE_FACE)])
        warp1 = self.warper.warp(bodyDetections[0][0])
        task = self.bodyAttributesEstimator.estimate(warp1, asyncEstimate=True)
        self.assertAsyncEstimation(task, BodyAttributes)
        task = self.bodyAttributesEstimator.estimateBatch([warp1] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, BodyAttributes)

    def test_aggregate_body_attributes(self):
        """
        Test body attributes aggregation.
        """
        estimations = self.estimate(SEVERAL_FACES)
        aggregated = self.bodyAttributesEstimator.aggregate(estimations)
        assert isinstance(aggregated, BodyAttributes)
        assert round(aggregated.apparentAge) == round(
            (sum([estimation.apparentAge for estimation in estimations])) / len(estimations)
        )

    def test_sleeve_correctness(self):
        """Sleeve length estimation correctness test"""
        cases = ((T_SHORT, SleeveLength.Short), (LONG_SLEEVE, SleeveLength.Long), (RED_EYES, SleeveLength.Unknown))
        for image, expectedLength in cases:
            with self.subTest(expectedLength):
                estimation = self.estimate(image)[0]
                assert expectedLength == estimation.sleeve.predominantState

    def test_backpack_correctness(self):
        """Backpack estimation correctness test"""
        cases = (
            (BACKPACK, BackpackStateEnum.Yes),
            (LONG_SLEEVE, BackpackStateEnum.No),
            (TURNED_HEAD_POSE_FACE, BackpackStateEnum.Unknown),
        )
        for image, expectedLength in cases:
            with self.subTest(expectedLength):
                estimation = self.estimate(image)[0]
                assert expectedLength == estimation.backpack.predominantState

    def test_headwear_correctness(self):
        """Headwear estimation correctness test"""
        cases = (
            (HOOD, HeadwearStateEnum.Yes),
            (LONG_SLEEVE, HeadwearStateEnum.No),
            (PALETTE_MODE, HeadwearStateEnum.Unknown),
        )
        for image, expectedLength in cases:
            with self.subTest(expectedLength):
                estimation = self.estimate(image)[0]
                assert expectedLength == estimation.headwear.predominantState

    def test_age_correctness(self):
        """Apparent age estimation correctness test"""

        estimation = self.estimate(LONG_SLEEVE)[0]
        assert 18 < estimation.apparentAge < 30

    def test_gender_correctness(self):
        """Apparent gender estimation correctness test"""
        cases = (
            (FROWNING, ApparentGenderEnum.Male),
            (LONG_SLEEVE, ApparentGenderEnum.Female),
            (BAD_THRESHOLD_WARP, ApparentGenderEnum.Unknown),
        )
        for image, expectedLength in cases:
            with self.subTest(expectedLength):
                estimation = self.estimate(image)[0]
                assert expectedLength == estimation.apparentGender.predominantGender

    def test_outwear_color_correctness(self):
        """Apparent gender estimation correctness test"""
        cases = (
            # (BEIGE, OutwearColorEnum.Beige),
            (BLACK, OutwearColorEnum.Black),
            (ANGER, OutwearColorEnum.Blue),
            # (BAD_THRESHOLD_WARP, OutwearColorEnum.Brown),
            (ANGER, OutwearColorEnum.Green),
            (RAISED, OutwearColorEnum.Grey),
            # (KHAKI, OutwearColorEnum.Khaki),
            # (COLORFUL, OutwearColorEnum.Multicolored),
            (SHAWL, OutwearColorEnum.Orange),
            # (PINK, OutwearColorEnum.Pink),
            (PINK, OutwearColorEnum.Purple),
            (RED, OutwearColorEnum.Red),
            (FULL_OCCLUDED_FACE, OutwearColorEnum.White),
            (YELLOW, OutwearColorEnum.Yellow),
        )
        for image, expectedLength in cases:
            with self.subTest(expectedLength):
                estimation = self.estimate(image)[0]
                print(image, estimation.outwearColor.colors)
                assert expectedLength in estimation.outwearColor.colors
