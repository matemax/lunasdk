"""Module contains a basic attributes estimator.

See `basic attributes`_.
"""
from enum import Enum
from typing import Union

from FaceEngine import IAttributeEstimatorPtr, AttributeRequest, AttributeResult  # pylint: disable=E0611,E0401
from FaceEngine import EthnicityEstimation, Ethnicity as CoreEthnicity  # pylint: disable=E0611,E0401

from lunavl.sdk.estimators.base_estimation import BaseEstimator, BaseEstimation
from lunavl.sdk.estimators.face_estimators.warper import Warp, WarpedImage


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
    def fromCoreEmotion(coreEthnicity: CoreEthnicity) -> 'Ethnicity':
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
            "predominant_ethnicity": str(self.predominateEmotion),
            "estimations": {
                "asian": self.asian,
                "indian": self.indian,
                "caucasian": self.caucasian,
                "african_american": self.africanAmerican
            }
        }

    @property
    def predominateEmotion(self) -> Ethnicity:
        """
        Get predominate ethnicity (ethnicity with max score value).

        Returns:
            ethnicity with max score value
        """
        return Ethnicity.fromCoreEmotion(self._coreEstimation.getPredominantEthnicity())


class BasicAttributes(BaseEstimation):
    """
    Class for basic attribute estimation

    Attributes:
        age (Optional[float]): age, number in range [0, 100]
        gender (Optional[float]): gender, number in range [0, 1]
        ethnicity (Optional[Ethnicities]): ethnicity
    """
    __slots__ = ("ethnicity", 'age', 'gender')

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

        if not coreEstimation.ethnicity_opt.isValid():
            self.age = None
        else:
            self.age = coreEstimation.age_opt.value()

        if not coreEstimation.gender_opt.isValid():
            self.gender = None
        else:
            self.gender = coreEstimation.gender_opt.value()

    def asDict(self) -> dict:
        """
        Convert to dict.

        Returns:
            dict with keys "ethnicity", "gender", "age"
        """
        res = {}
        if self.ethnicity is not None:
            res["ethnicities"] = self.ethnicity.asDict()
        else:
            res["ethnicities"] = None
        if self.age is not None:
            res["age"] = round(self.age)
        else:
            res["age"] = None
        if self.gender is not None:
            res["gender"] = round(self.gender)
        else:
            res["gender"] = None
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
    def estimate(self, warp: Union[Warp, WarpedImage], estimateAge: bool, estimateGender: bool,
                 estimateEthnicity: bool) -> BasicAttributes:
        """
        Estimate ethnicity.

        Args:
            warp: warped image
            estimateAge: estimate age or not
            estimateGender: estimate gender or not
            estimateEthnicity: estimate ethnicity or not

        Returns:
            estimated ethnicity
        """
        dtAttributes = 0
        if estimateAge:
            dtAttributes |= AttributeRequest.estimateAge
        if estimateGender:
            dtAttributes |= AttributeRequest.estimateGender
        if estimateEthnicity:
            dtAttributes |= AttributeRequest.estimateEthnicity

        error, baseAttributes = self._coreEstimator.estimate(warp.warpedImage.coreImage,
                                                             AttributeRequest(dtAttributes))
        if error.isError:
            raise ValueError("12343")
        return BasicAttributes(baseAttributes)
