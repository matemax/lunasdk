"""Module contains an emotion estimator

See emotions_.
"""
from enum import Enum
from typing import List, Literal, Union, overload

from FaceEngine import Emotions as CoreEmotions  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation

from ...async_task import AsyncTask, DefaultPostprocessingFactory
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage


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
    Container for storing estimated emotions. List of emotions is represented in enum Emotion. Each emotion
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


POST_PROCESSING = DefaultPostprocessingFactory(Emotions)


class EmotionsEstimator(BaseEstimator):
    """
    Emotions estimator.
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: Literal[False] = False) -> Emotions:
        ...

    @overload
    def estimate(self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: Literal[True]) -> AsyncTask[Emotions]:
        ...

    def estimate(
        self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: bool = False
    ) -> Union[Emotions, AsyncTask[Emotions]]:
        """
        Estimate emotion on warp.

        Args:
            warp: warped image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated emotions if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(warp.warpedImage.coreImage)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, emotions = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        return POST_PROCESSING.postProcessing(error, emotions)

    #  pylint: disable=W0221
    def estimateBatch(
        self, warps: List[Union[FaceWarp, FaceWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[Emotions], AsyncTask[List[Emotions]]]:
        """
        Batch estimate emotions

        Args:
            warps:warped images
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated emotions if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        coreImages = [warp.warpedImage.coreImage for warp in warps]

        validateInputByBatchEstimator(self._coreEstimator, coreImages)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        error, masks = self._coreEstimator.estimate(coreImages)
        return POST_PROCESSING.postProcessingBatch(error, masks)
