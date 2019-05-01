"""
Module contains an emotion estimator
"""
from enum import Enum
from typing import Union

from FaceEngine import IEmotionsEstimatorPtr, Emotions as CoreEmotions  # pylint: disable=E0611,E0401

from lunavl.sdk.faceengine.warper import Warp, WarpedImage


class Emotion(Enum):
    """
    Emotions enum
    """
    #: Anger
    Anger = 1
    #: Disgust
    Disgust = 2
    #: Fear
    Fear = 3
    #: Happiness
    Happiness = 4
    #: Neutral
    Neutral = 5
    #: Sadness
    Sadness = 6
    #: Surprise
    Surprise = 7

    @staticmethod
    def fromCoreEmotion(coreEmotion: CoreEmotions) -> 'Emotion':
        """
        Get enum element by core emotion.

        Args:
            coreEmotion:

        Returns:
            corresponding emotion
        """
        return getattr(Emotion, coreEmotion.name)


class Emotions:
    """
    Container for storing estimate emotions. List of emotions is represented in enum Emotion. Each emotion
    is characterized a score (value in range [0,1]). Sum of all scores is equal to 1. Predominate
    emotion is emotion with max value of score

    Attributes:
        _coreEmotions: core estimation
    """
    __slots__ = ['_coreEmotions']

    def __init__(self, coreEmotions):
        """
        Init.

        Args:
            coreEmotions:  estimation from core
        """
        self._coreEmotions = coreEmotions

    def asDict(self):
        """
        Convert estimation to dict.

        Returns:
            dict with keys 'predominate_emotion' and 'estimations'
        """
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

    def __repr__(self) -> str:
        """
        Representation.

        Returns:
            str(self.asDict())
        """
        return str(self.asDict())

    @property
    def anger(self) -> float:
        """
        Get anger emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEmotions.anger

    @property
    def disgust(self):
        """
        Get disgust emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEmotions.disgust

    @property
    def fear(self):
        """
        Get fear emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEmotions.fear

    @property
    def happiness(self):
        """
        Get happiness emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEmotions.happiness

    @property
    def sadness(self):
        """
        Get sadness emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEmotions.sadness

    @property
    def surprise(self):
        """
        Get surprise emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEmotions.surprise

    @property
    def neutral(self):
        """
        Get neutral emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEmotions.neutral

    @property
    def predominateEmotion(self) -> Emotion:
        """
        Get predominate emotion (emotion with max score value).

        Returns:
            emotion with max score value
        """
        return Emotion.fromCoreEmotion(self._coreEmotions.getPredominantEmotion())


class EmotionsEstimator:
    """
    Emotions estimator.

    Attributes:
        _coreEstimator (IEmotionsEstimatorPtr): core estimator
    """
    __slots__ = ['_coreEstimator']

    def __init__(self, coreEstimator: IEmotionsEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        self._coreEstimator = coreEstimator

    def estimate(self, warp: Union[Warp, WarpedImage]) -> Emotions:
        """
        Estimate emotion on warp.

        Args:
            warp: warped image

        Returns:
            estimated emotions
        """
        error, emotions = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        if error.isError:
            raise ValueError("12343")
        return Emotions(emotions)
