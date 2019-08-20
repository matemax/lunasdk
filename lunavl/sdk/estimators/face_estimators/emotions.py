"""Module contains an emotion estimator

See emotions_.
"""
from enum import Enum
from typing import Union

from FaceEngine import IEmotionsEstimatorPtr, Emotions as CoreEmotions  # pylint: disable=E0611,E0401
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWarp, LunaSDKException

from lunavl.sdk.estimators.base_estimation import BaseEstimation, BaseEstimator
from lunavl.sdk.estimators.face_estimators.warper import Warp, WarpedImage


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
    def fromCoreEmotion(coreEmotion: CoreEmotions) -> "Emotion":
        """
        Get enum element by core emotion.

        Args:
            coreEmotion:

        Returns:
            corresponding emotion
        """
        return getattr(Emotion, coreEmotion.name)


class Emotions(BaseEstimation):
    """
    Container for storing estimate emotions. List of emotions is represented in enum Emotion. Each emotion
    is characterized a score (value in range [0,1]). Sum of all scores is equal to 1. Predominate
    emotion is emotion with max value of score.

    Estimation properties:

        - anger
        - disgust
        - fear
        - happiness
        - sadness
        - surprise
        - neutral
        - predominateEmotion

    """

    #  pylint: disable=W0235
    def __init__(self, coreEmotions):
        """
        Init.

        Args:
            coreEmotions:  estimation from core
        """
        super().__init__(coreEmotions)

    def asDict(self):
        """
        Convert estimation to dict.

        Returns:
            dict with keys 'predominate_emotion' and 'estimations'
        """
        return {
            "predominant_emotion": self.predominateEmotion.name.lower(),
            "estimations": {
                "anger": self.anger,
                "disgust": self.disgust,
                "fear": self.fear,
                "happiness": self.happiness,
                "sadness": self.sadness,
                "surprise": self.surprise,
                "neutral": self.neutral,
            },
        }

    @property
    def anger(self) -> float:
        """
        Get anger emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.anger

    @property
    def disgust(self):
        """
        Get disgust emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.disgust

    @property
    def fear(self):
        """
        Get fear emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.fear

    @property
    def happiness(self):
        """
        Get happiness emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.happiness

    @property
    def sadness(self):
        """
        Get sadness emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.sadness

    @property
    def surprise(self):
        """
        Get surprise emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.surprise

    @property
    def neutral(self):
        """
        Get neutral emotion value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.neutral

    @property
    def predominateEmotion(self) -> Emotion:
        """
        Get predominate emotion (emotion with max score value).

        Returns:
            emotion with max score value
        """
        return Emotion.fromCoreEmotion(self._coreEstimation.getPredominantEmotion())


class EmotionsEstimator(BaseEstimator):
    """
    Emotions estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: IEmotionsEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    @CoreExceptionWarp(LunaVLError.EstimationEmotionsError)
    def estimate(self, warp: Union[Warp, WarpedImage]) -> Emotions:
        """
        Estimate emotion on warp.

        Args:
            warp: warped image

        Returns:
            estimated emotions
        Raises:
            LunaSDKException: if estimation failed
        """
        error, emotions = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return Emotions(emotions)
