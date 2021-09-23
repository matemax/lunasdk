"""
Module realize face descriptor match.

see `face descriptors matching`_.
"""
from typing import List, Union

from FaceEngine import IDescriptorMatcherPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.descriptors.descriptors import FaceDescriptorFactory
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


class FaceMatcher:
    """
    Base estimator class. Class is  a container for core estimations. Mostly estimate attributes  can be get through
    a corresponding properties.

    Attributes:
        _coreMatcher (IDescriptorMatcherPtr): core matcher
        descriptorFactory (FaceDescriptorFactory): face descriptor factory
    """

    __slots__ = ("_coreMatcher", "descriptorFactory")

    def __init__(self, coreMatcher: IDescriptorMatcherPtr, descriptorFactory: FaceDescriptorFactory):
        """
        Init.

        Args:
            coreMatcher: core matcher
        """
        self._coreMatcher: IDescriptorMatcherPtr = coreMatcher
        self.descriptorFactory: FaceDescriptorFactory = descriptorFactory

    def match(
        self,
        reference: Union[FaceDescriptor, bytes],
        candidates: Union[FaceDescriptor, bytes, List[Union[FaceDescriptor, bytes]], FaceDescriptorBatch],
    ) -> Union[MatchingResult, List[MatchingResult]]:
        """
        Match face descriptor vs face descriptors.

        Returns:
            List of matching results if match by several descriptors otherwise one MatchingResult.
        """
        if isinstance(reference, bytes):
            referenceForMatcher = self.descriptorFactory.generateDescriptor(reference)
        else:
            referenceForMatcher = reference

        if isinstance(candidates, bytes):
            candidatesForMatcher = self.descriptorFactory.generateDescriptor(candidates)
        elif isinstance(candidates, list):
            candidatesForMatcher = []
            for idx in range(len(candidates)):
                if isinstance(candidates[idx], bytes):
                    candidatesForMatcher.append(self.descriptorFactory.generateDescriptor(candidates[idx]))
                else:
                    candidatesForMatcher.append(candidates[idx])
        else:
            candidatesForMatcher = candidates

        if isinstance(candidatesForMatcher, FaceDescriptor):
            error, matchResults = self._coreMatcher.match(
                referenceForMatcher.coreEstimation, candidatesForMatcher.coreEstimation
            )
        elif isinstance(candidatesForMatcher, FaceDescriptorBatch):
            error, matchResults = self._coreMatcher.match(
                referenceForMatcher.coreEstimation, candidatesForMatcher.coreEstimation
            )
        else:
            batch = self.descriptorFactory.generateDescriptorsBatch(len(candidatesForMatcher))
            for candidate in candidatesForMatcher:
                batch.append(candidate)
            error, matchResults = self._coreMatcher.match(referenceForMatcher.coreEstimation, batch.coreEstimation)

        assertError(error)
        return matchResults
