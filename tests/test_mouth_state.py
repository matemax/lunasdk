from collections import namedtuple

import jsonschema

from lunavl.sdk.estimators.face_estimators.mouth_state import MouthStateEstimator, MouthStates
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.image_utils.image import VLImage
from tests.detect_test_class import FaceDetectTestClass
from tests.resources import CLEAN_ONE_FACE, WARP_WHITE_MAN
from tests.schemas import MOUTH_STATES_SCHEMA

VLIMAGE_ONE_FACE = VLImage.load(filename=CLEAN_ONE_FACE)


class TestMouthEstimation(FaceDetectTestClass):
    """
    Test Mouth States Estimation
    """

    # mouth state estimator
    mouthEstimator: MouthStateEstimator

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.warper = cls.faceEngine.createWarper()
        cls.mouthEstimator = cls.faceEngine.createMouthEstimator()
        CaseWarp = namedtuple("CaseWarp", ("warp", "detector"))
        cls.warpList = []
        for detector in cls.detectors:
            detection = detector.detectOne(VLIMAGE_ONE_FACE)
            cls.warpList.append(CaseWarp(cls.warper.warp(detection), detector.detectorType.name))
        cls.warpList.append(CaseWarp(FaceWarpedImage(VLImage.load(filename=WARP_WHITE_MAN)), "None"))

    def test_mouth_states_with_different_type(self):
        """
        Test estimate mouth state with warp and warpedImage
        """
        for case in self.warpList:
            with self.subTest(warp=type(case.warp).__name__, detectorType=case.detector):
                mouthStates = self.mouthEstimator.estimate(case.warp)
                assert isinstance(mouthStates, MouthStates), f"{mouthStates.__class__} is not {MouthStates}"
                for attr in ("smile", "mouth", "occlusion"):
                    mouthState = getattr(mouthStates, f"{attr}")
                    assert isinstance(mouthState, float), f"{attr} is not float"
                    assert 0 <= mouthState <= 1, f"{attr} out of range [0,1]"

    def test_mouth_estimation_as_dict(self):
        """
        Test mouth states convert to dict
        """
        for case in self.warpList:
            with self.subTest(warp=type(case.warp).__name__, detectorType=case.detector):
                emotionDict = self.mouthEstimator.estimate(case.warp).asDict()
                assert (
                    jsonschema.validate(emotionDict, MOUTH_STATES_SCHEMA) is None
                ), f"Mouth states: {emotionDict} does not match with schema: {MOUTH_STATES_SCHEMA}"
