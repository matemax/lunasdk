from typing import Optional

from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.detect_test_class import DetectTestClass
from tests.resources import EMOTION_FACES, ALL_EMOTIONS

EMOTION_IMAGES = {emotion: VLImage.load(filename=imagePath) for emotion, imagePath in EMOTION_FACES.items()}


class TestEstimateEmotions(DetectTestClass):
    """
    Test estimate emotions.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.defaultDetector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)
        cls.warper = cls.faceEngine.createWarper()
        cls.emotionEstimator = cls.faceEngine.createEmotionEstimator()

    def assert_emotion_reply(self, emotionDetection: dict, predominantEmotion: Optional[str] = None) -> None:
        """
        Assert emotions dict
        Args:
            emotionDetection: dict with predominant and all emotions
            predominantEmotion: expected predominant emotion
        """
        self.validate_emotion_dict(emotionDetection)
        if predominantEmotion is not None:
            assert emotionDetection["predominant_emotion"], predominantEmotion

    @staticmethod
    def validate_emotion_dict(receivedDict: dict):
        """
        Validate emotion dict
        """
        assert sorted(["predominant_emotion", "estimations"]), sorted(receivedDict.keys())
        assert sorted(ALL_EMOTIONS), receivedDict["estimations"]
        for emotion, emotionValue in receivedDict["estimations"].items():
            assert 0 < emotionValue < 1

    def test_estimate_emotions_as_dict(self):
        """
        Test emotion estimator 'asDict' method
        """
        for detector in self.detectors:
            with self.subTest(detectorType=detector.detectorType):
                faceDetection = detector.detectOne(EMOTION_IMAGES["fear"])
                warp = self.warper.warp(faceDetection)
                emotionDict = self.emotionEstimator.estimate(warp.warpedImage).asDict()
                self.validate_emotion_dict(emotionDict)

    def test_estimate_emotions(self):
        """
        Test all emotions estimations
        """
        for detector in self.detectors:
            for emotion in ALL_EMOTIONS:
                with self.subTest(detectorType=detector.detectorType, emotion=emotion):
                    faceDetection = detector.detectOne(EMOTION_IMAGES[emotion])
                    warp = self.warper.warp(faceDetection)
                    emotionDict = self.emotionEstimator.estimate(warp.warpedImage).asDict()
                    self.assert_emotion_reply(emotionDict, emotion)

    def test_estimate_emotion_not_warped(self):
        """
        Test emotions with not warped image
        """
        for detector in self.detectors:
            for emotion in ALL_EMOTIONS:
                with self.subTest(detectorType=detector.detectorType, emotion=emotion):
                    faceDetection = detector.detectOne(EMOTION_IMAGES[emotion])
                    warp = self.warper.warp(faceDetection)
                    VLImage
                    emotionDict = self.emotionEstimator.estimate(warp.warpedImage).asDict()
                    self.assert_emotion_reply(emotionDict, emotion)
