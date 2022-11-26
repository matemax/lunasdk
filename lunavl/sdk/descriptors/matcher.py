"""
Module realize face descriptor match.

see `face descriptors matching`_.
"""
from typing import List, Union, Generic

from FaceEngine import IDescriptorMatcherPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.descriptors.descriptors import (
    BaseDescriptor,
    BaseDescriptorFactory,
    DescriptorType,
    BodyDescriptor,
)
from lunavl.sdk.errors.exceptions import assertError
from lunavl.sdk.estimators.face_estimators.face_descriptor import FaceDescriptor, FaceDescriptorBatch


class MatchingResult:
    """
    Structure for storing matching results.

    Attributes:
        distance (float): L2 distance between descriptors
        similarity (float): descriptor similarity [0..1]
    """

    __slots__ = ("distance", "similarity")

    def __init__(self, distance: float, similarity: float):
        self.distance = distance
        self.similarity = similarity


class BaseMatcher(Generic[DescriptorType]):
    """
    Base estimator class. Class is  a container for core estimations. Mostly estimate attributes  can be get through
    a corresponding properties.

    Attributes:
        _coreMatcher (IDescriptorMatcherPtr): core matcher
        descriptorFactory (BaseDescriptorFactory): descriptor factory
    """

    __slots__ = ("_coreMatcher", "descriptorFactory")

    def __init__(self, coreMatcher: IDescriptorMatcherPtr, descriptorFactory: BaseDescriptorFactory[DescriptorType]):
        """
        Init.

        Args:
            coreMatcher: core matcher
        """
        self._coreMatcher: IDescriptorMatcherPtr = coreMatcher
        self.descriptorFactory: BaseDescriptorFactory[DescriptorType] = descriptorFactory

    def match(
        self,
        reference: Union[DescriptorType, bytes],
        candidates: Union[DescriptorType, bytes, List[Union[DescriptorType, bytes]], FaceDescriptorBatch],
    ) -> Union[MatchingResult, List[MatchingResult]]:
        """
        Match face/body descriptor vs face/body descriptors.

        Returns:
            List of matching results if match by several descriptors otherwise one MatchingResult.
        Raises:
            TypeError: if candidates has incorrect type
        """
        if isinstance(reference, bytes):
            referenceForMatcher = self.descriptorFactory.generateDescriptor(reference)
        else:
            referenceForMatcher = reference

        if isinstance(candidates, bytes):
            candidatesForMatcher = self.descriptorFactory.generateDescriptor(candidates)
            error, matchResults = self._coreMatcher.match(
                referenceForMatcher.coreEstimation, candidatesForMatcher.coreEstimation
            )
        elif isinstance(candidates, BaseDescriptor):
            error, matchResults = self._coreMatcher.match(referenceForMatcher.coreEstimation, candidates.coreEstimation)
        elif isinstance(candidates, list):
            batch = self.descriptorFactory.generateDescriptorsBatch(len(candidates))
            for cand in candidates:
                if isinstance(cand, bytes):
                    candidate = self.descriptorFactory.generateDescriptor(cand)
                else:
                    candidate = cand
                batch.append(candidate)
            error, matchResults = self._coreMatcher.match(referenceForMatcher.coreEstimation, batch.coreEstimation)
        elif isinstance(candidates, FaceDescriptorBatch):
            error, matchResults = self._coreMatcher.match(referenceForMatcher.coreEstimation, candidates.coreEstimation)
        else:
            raise TypeError(f"Bad candidates type: {type(candidates)}")

        assertError(error)
        return matchResults


class FaceMatcher(BaseMatcher[FaceDescriptor]):
    pass


class BodyMatcher(BaseMatcher[BodyDescriptor]):
    pass
