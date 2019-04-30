from enum import Enum
from typing import Union

from FaceEngine import IEmotionsEstimatorPtr, Emotions as CoreEmotions  # pylint: disable=E0611,E0401

from lunavl.sdk.faceengine.warper import Warp, WarpedImage


class Emotion(Enum):
    Anger = 1
    Disgust = 2
    Fear = 3
    Happiness = 4
    Neutral = 5
    Sadness = 6
    Surprise = 7

    @staticmethod
    def fromCoreEmotion(coreEmotion: CoreEmotions) -> 'Emotion':
        return getattr(Emotion, coreEmotion.name)


class Emotions:
    __slots__ = ['_coreEmotions']

    def __init__(self, coreEmotions):
        self._coreEmotions = coreEmotions

    def asDict(self):
        return {'predominate_emotion': self.predominateEmotion.name.lower(),
                'estimations': {
                    'anger': self.anger,
                    'disgust': self.disgust,
                    'fear': self.fear,
                    'happiness': self.happiness,
                    'sadness': self.sadness,
                    'surprise': self.surprise,
                    'neutral': self.neutral,
                }}

    def __repr__(self):
        return str(self.asDict())

    @property
    def anger(self):
        return self._coreEmotions.anger

    @property
    def disgust(self):
        return self._coreEmotions.disgust

    @property
    def fear(self):
        return self._coreEmotions.fear

    @property
    def happiness(self):
        return self._coreEmotions.happiness

    @property
    def sadness(self):
        return self._coreEmotions.sadness

    @property
    def surprise(self):
        return self._coreEmotions.surprise

    @property
    def neutral(self):
        return self._coreEmotions.neutral

    @property
    def predominateEmotion(self) -> Emotion:
        return Emotion.fromCoreEmotion(self._coreEmotions.getPredominantEmotion())


class EmotionsEstimator:
    __slots__ = ['_coreEstimator']

    def __init__(self, coreEstimator: IEmotionsEstimatorPtr):
        self._coreEstimator = coreEstimator

    def estimate(self, warp: Union[Warp, WarpedImage]):
        error, emotions = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        if error.isError:
            raise ValueError("12343")
        return Emotions(emotions)
