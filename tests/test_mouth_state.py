import jsonschema

from lunavl.sdk.estimators.face_estimators.mouth_state import MouthStateEstimator, MouthStates
from lunavl.sdk.estimators.face_estimators.warper import WarpedImage
from lunavl.sdk.image_utils.image import VLImage
from tests.detect_test_class import DetectTestClass
from tests.resources import CLEAN_ONE_FACE, WARP_WHITE_MAN
from tests.schemas import MOUTH_STATES

VLIMAGE_ONE_FACE = VLImage.load(filename=CLEAN_ONE_FACE)
WARPED_IMAGE = WarpedImage(VLImage.load(filename=WARP_WHITE_MAN))


class TestMouthEstimation(DetectTestClass):
    """
    Test Mouth States Estimation
    """

    mouthEstimator: MouthStateEstimator = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.warper = cls.faceEngine.createWarper()
        cls.mouthEstimator = cls.faceEngine.createMouthEstimator()
        cls.warpList = []
        for detector in cls.detectors:
            detection = detector.detectOne(VLIMAGE_ONE_FACE)
            cls.warpList.append(cls.warper.warp(detection))

    def test_mouth_states_with_warp_structure(self):
        """
        Test estimated states on warp structure
        """
        for warp in self.warpList:
            with self.subTest(warp=warp):
                mouthStates = self.mouthEstimator.estimate(warp)
                assert isinstance(mouthStates, MouthStates), f"{mouthStates.__class__} is not {MouthStates}"
                assert all(
                    isinstance(getattr(mouthStates, f"{mouthState}"), float)
                    for mouthState in ("smile", "mouth", "occlusion")
                )

    def test_mouth_states_with_warped_image(self):
        """
        Test estimated states on warpedImage
        """
        mouthStates = self.mouthEstimator.estimate(WARPED_IMAGE)
        assert isinstance(mouthStates, MouthStates), f"{mouthStates.__class__} is not {MouthStates}"
        assert all(
            isinstance(getattr(mouthStates, f"{mouthState}"), float) for mouthState in ("smile", "mouth", "occlusion")
        )

    def test_mouth_estimation_as_dict(self):
        """
        Test mouth states convert to dict
        """
        for warp in self.warpList:
            with self.subTest(warp=warp):
                emotionDict = self.mouthEstimator.estimate(warp).asDict()
                assert (
                    jsonschema.validate(emotionDict, MOUTH_STATES) is None
                ), f"Mouth states: {emotionDict} does not match with schema: {MOUTH_STATES}"
