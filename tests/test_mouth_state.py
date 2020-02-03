import jsonschema

from lunavl.sdk.estimators.face_estimators.mouth_state import MouthStateEstimator, MouthStates, SmileEstimation
from lunavl.sdk.estimators.face_estimators.warper import Warper
from lunavl.sdk.image_utils.image import VLImage
from tests.detect_test_class import DetectTestClass
from tests.resources import CLEAN_ONE_FACE
from tests.schemas import MOUTH_STATES

VLIMAGE_ONE_FACE = VLImage.load(filename=CLEAN_ONE_FACE)


class TestMouthEstimation(DetectTestClass):
    warper: Warper = None
    mouthEstimator: MouthStateEstimator = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.warper = cls.faceEngine.createWarper()
        cls.mouthEstimator = cls.faceEngine.createMouthEstimator()

    def test_common_mouth_states_with_different_method(self):
        """
        Test estimated states on warpedImage
        """
        for detector in self.detectors:
            for method in ("warp", "warpedImage"):
                with self.subTest(estimateMethod=method, detectorType=detector.detectorType):
                    detection = detector.detectOne(VLIMAGE_ONE_FACE)
                    warp = self.warper.warp(detection)
                    if method == "warpedImage":
                        mouthStates = self.mouthEstimator.estimate(warp.warpedImage)
                    else:
                        mouthStates = self.mouthEstimator.estimate(warp)
                    assert isinstance(mouthStates, MouthStates), f"{mouthStates.__class__} is not {MouthStates}"
                    assert all(
                        isinstance(getattr(mouthStates, f"{mouthState}"), float)
                        for mouthState in ("smile", "mouth", "occlusion")
                    )

    def test_mouth_estimation_as_dict(self):
        """
        Test mouth states convert to dict
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                detection = detector.detectOne(VLIMAGE_ONE_FACE)
                warp = self.warper.warp(detection)
                emotionDict = self.mouthEstimator.estimate(warp).asDict()
                assert (
                    jsonschema.validate(emotionDict, MOUTH_STATES) is None
                ), f"Mouth states: {emotionDict} does not match with schema: {MOUTH_STATES}"
