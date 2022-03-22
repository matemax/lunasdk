"""Module contains an eyebrow expression estimator

See eyebrow_expression_.
"""
from enum import Enum
from typing import Union, List

from FaceEngine import (
    EyeBrowEstimation as CoreEyeBrowEstimation,
    EyeBrowState as CoreEyebrowState,
)  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage
from ...async_task import AsyncTask, DefaultPostprocessingFactory


class EyebrowExpression(Enum):
    """
    Eyebrow expression enum
    """

    #: Neutral
    Neutral = 1
    #: Raised
    Raised = 2
    #: Squinting
    Squinting = 3
    #: Frowning
    Frowning = 4

    @staticmethod
    def fromCoreEyebrow(coreEyebrown: CoreEyebrowState) -> "EyebrowExpression":
        """
        Get enum element by core eyebrow state.

        Args:
            coreEyebrow: core eyebrow expression

        Returns:
            corresponding eyebrow state
        """
        return getattr(EyebrowExpression, coreEyebrown.name)


class EyebrowExpressions(BaseEstimation):
    """
    Container for storing a estimated eyebrow expression. List of eyebrow expressions is represented in enum
    EyebrowExpression. Each eyebrow expression is characterized a score (value in range [0,1]). Sum of all scores is
    equal to 1. A predominate expression is with max value of score.

    Estimation properties:

        - neutral
        - raised
        - squinting
        - frowning

    """

    #  pylint: disable=W0235
    def __init__(self, coreEyeBrow: CoreEyeBrowEstimation):
        """
        Init.

        Args:
            coreEyeBrow:  eyebrow expression from core
        """
        super().__init__(coreEyeBrow)

    def asDict(self):
        """
        Convert estimation to dict.

        Returns:
            dict with keys 'predominant_expression' and 'estimations'
        """
        return {
            "predominant_expression": self.predominateExpression.name.lower(),
            "estimations": {
                "neutral": self.neutral,
                "raised": self.raised,
                "squinting": self.squinting,
                "frowning": self.frowning,
            },
        }

    @property
    def neutral(self) -> float:
        """
        Get neutral expression value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.Neutral

    @property
    def raised(self):
        """
        Get raised expression value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.Raised

    @property
    def frowning(self):
        """
        Get frowning expression value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.Frowning

    @property
    def squinting(self):
        """
        Get squinting expression value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.Squinting

    @property
    def predominateExpression(self) -> EyebrowExpression:
        """
        Get predominate eyebrow expression (expression with max score value).

        Returns:
            eyebrow expression with max score value
        """
        return EyebrowExpression.fromCoreEyebrow(self._coreEstimation.EyeBrowState)


POST_PROCESSING = DefaultPostprocessingFactory(EyebrowExpressions)


class EyebrowExpressionEstimator(BaseEstimator):
    """
    Eyebrow expression estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    def estimate(
        self, warp: Union[FaceWarp, FaceWarpedImage], asyncEstimate: bool = False
    ) -> Union[EyebrowExpressions, AsyncTask[EyebrowExpressions]]:
        """
        Eyebrow expression on warp.

        Args:
            warp: warped image
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated Eyebrow expression if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(warp.warpedImage.coreImage)
            return AsyncTask(task, POST_PROCESSING.postProcessing)
        error, eyebrow = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        return POST_PROCESSING.postProcessing(error, eyebrow)

    #  pylint: disable=W0221
    def estimateBatch(
        self, warps: List[Union[FaceWarp, FaceWarpedImage]], asyncEstimate: bool = False
    ) -> Union[List[EyebrowExpressions], AsyncTask[List[EyebrowExpressions]]]:
        """
        Batch estimate eyebrow expressions

        Args:
            warps:warped images
            asyncEstimate: estimate or run estimation in background
        Returns:
            list of estimated Eyebrow expressions if asyncEstimate is false otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        coreImages = [warp.warpedImage.coreImage for warp in warps]

        validateInputByBatchEstimator(self._coreEstimator, coreImages)
        if asyncEstimate:
            task = self._coreEstimator.asyncEstimate(coreImages)
            return AsyncTask(task, POST_PROCESSING.postProcessingBatch)
        error, estimations = self._coreEstimator.estimate(coreImages)
        return POST_PROCESSING.postProcessingBatch(error, estimations)
