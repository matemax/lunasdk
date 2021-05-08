"""Module contains a basic attributes estimator.

See `basic attributes`_.
"""
from enum import Enum
from typing import Union, Dict, Any, List, Tuple

from FaceEngine import (
    IAttributeEstimatorPtr,
    AttributeRequest,
    AttributeResult,
    EthnicityEstimation,
    Ethnicity as CoreEthnicity,
)  # pylint: disable=E0611,E0401; pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException
from lunavl.sdk.base import BaseEstimation
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import validateInputByBatchEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage


class Ethnicity(Enum):
    """
    Enum for ethnicities.
    """

    #: african american
    AfricanAmerican = 1
    #: asian
    Asian = 2
    #: indian
    Indian = 3
    #: caucasian
    Caucasian = 4

    @staticmethod
    def fromCoreEthnicity(coreEthnicity: CoreEthnicity) -> "Ethnicity":
        """
        Get enum element by core ethnicity.

        Args:
            coreEthnicity: core ethnicity

        Returns:
            corresponding ethnicity
        """
        return getattr(Ethnicity, coreEthnicity.name)

    def __str__(self):
        """
        Convert enum element to string.

        Returns:
            snake case ethnicity
        """
        if self in (Ethnicity.Asian, Ethnicity.Indian, Ethnicity.Caucasian):
            # pylint: disable=E1101
            return self.name.lower()
        return "african_american"


class Ethnicities(BaseEstimation):
    """
    Class for ethnicities estimation.

    Estimation properties:

        - asian
        - indian
        - caucasian
        - africanAmerican
        - predominateEmotion
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: EthnicityEstimation):
        """
        Init.

        Args:
            coreEstimation: core ethnicities estimation
        """
        super().__init__(coreEstimation)

    @property
    def asian(self) -> float:
        """
        Get asian ethnicity value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.asian

    @property
    def indian(self):
        """
        Get indian ethnicity value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.indian

    @property
    def caucasian(self):
        """
        Get caucasian ethnicity value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.caucasian

    @property
    def africanAmerican(self):
        """
        Get african american ethnicity value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.africanAmerican

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            dict in platform format
        """
        return {
            "predominant_ethnicity": str(self.predominantEthnicity),
            "estimations": {
                "asian": self.asian,
                "indian": self.indian,
                "caucasian": self.caucasian,
                "african_american": self.africanAmerican,
            },
        }

    @property
    def predominantEthnicity(self) -> Ethnicity:
        """
        Get predominate ethnicity (ethnicity with max score value).

        Returns:
            ethnicity with max score value
        """
        return Ethnicity.fromCoreEthnicity(self._coreEstimation.getPredominantEthnicity())


class BasicAttributes(BaseEstimation):
    """
    Class for basic attribute estimation

    Attributes:
        age (Optional[float]): age, number in range [0, 100]
        gender (Optional[float]): gender, number in range [0, 1]
        ethnicity (Optional[Ethnicities]): ethnicity
    """

    __slots__ = ("ethnicity", "age", "gender")

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: AttributeResult):
        """
        Init.

        Args:
            coreEstimation: core ethnicity estimation
        """
        super().__init__(coreEstimation)

        if not coreEstimation.ethnicity_opt.isValid():
            self.ethnicity = None
        else:
            self.ethnicity = Ethnicities(coreEstimation.ethnicity_opt.value())

        if not coreEstimation.age_opt.isValid():
            self.age = None
        else:
            self.age = coreEstimation.age_opt.value()

        if not coreEstimation.gender_opt.isValid():
            self.gender = None
        else:
            self.gender = coreEstimation.gender_opt.value()

    def asDict(self) -> Dict[str, Any]:
        """
        Convert to dict.

        Returns:
            dict with keys "ethnicity", "gender", "age"
        """
        res = {
            "ethnicities": self.ethnicity.asDict() if self.ethnicity is not None else None,
            "age": round(self.age) if self.age is not None else None,
            "gender": round(self.gender) if self.gender is not None else None,
        }
        return res


class BasicAttributesEstimator(BaseEstimator):
    """
    Basic attributes estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: IAttributeEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    @CoreExceptionWrap(LunaVLError.EstimationBasicAttributeError)
    def estimate(
        self, warp: Union[FaceWarp, FaceWarpedImage], estimateAge: bool, estimateGender: bool, estimateEthnicity: bool
    ) -> BasicAttributes:
        """
        Estimate a basic attributes (age, gender, ethnicity) from warped images.

        Args:
            warp: warped image
            estimateAge: estimate age or not
            estimateGender: estimate gender or not
            estimateEthnicity: estimate ethnicity or not

        Returns:
            estimated age, gender, ethnicity
        Raises:
            LunaSDKException: if estimation failed
        """
        dtAttributes = 0
        if estimateAge:
            dtAttributes |= AttributeRequest.estimateAge
        if estimateGender:
            dtAttributes |= AttributeRequest.estimateGender
        if estimateEthnicity:
            dtAttributes |= AttributeRequest.estimateEthnicity

        error, baseAttributes = self._coreEstimator.estimate(warp.warpedImage.coreImage, AttributeRequest(dtAttributes))
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return BasicAttributes(baseAttributes)

    @CoreExceptionWrap(LunaVLError.BatchEstimationBasicAttributeError)
    def estimateBasicAttributesBatch(
        self,
        warps: List[Union[FaceWarp, FaceWarpedImage]],
        estimateAge: bool,
        estimateGender: bool,
        estimateEthnicity: bool,
        aggregate: bool = False,
    ) -> Tuple[List[BasicAttributes], Union[None, BasicAttributes]]:
        """
        Batch basic attributes estimation on warped images.

        Args:
            warps: warped images
            estimateAge: estimate age or not
            estimateGender: estimate gender or not
            estimateEthnicity: estimate ethnicity or not
            aggregate:  aggregate attributes to one or not

        Returns:
            tuple, first element - list estimated attributes in corresponding order,
            second - optional aggregated attributes.
        Raises:
            LunaSDKException: if estimation failed
        """
        dtAttributes = 0
        if estimateAge:
            dtAttributes |= AttributeRequest.estimateAge
        if estimateGender:
            dtAttributes |= AttributeRequest.estimateGender
        if estimateEthnicity:
            dtAttributes |= AttributeRequest.estimateEthnicity

        images = [warp.warpedImage.coreImage for warp in warps]

        validateInputByBatchEstimator(self._coreEstimator, images, AttributeRequest(dtAttributes))
        error, baseAttributes, aggregateAttribute = self._coreEstimator.estimate(images, AttributeRequest(dtAttributes))
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

        attributes = [BasicAttributes(baseAttribute) for baseAttribute in baseAttributes]
        if aggregate:
            return attributes, BasicAttributes(aggregateAttribute)
        else:
            return attributes, None
