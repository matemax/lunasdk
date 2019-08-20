"""
Module realize face descriptor match.

see `face descriptors matching`_.
"""
from abc import abstractmethod
from typing import List, Union

from FaceEngine import IDescriptorMatcherPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.face_descriptor import FaceDescriptor, FaceDescriptorBatch
from lunavl.sdk.faceengine.descriptors import FaceDescriptorFactory


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
        self._coreMatcher = coreMatcher
        self.descriptorFactory = descriptorFactory

    @abstractmethod
    def match(
        self, reference: FaceDescriptor, candidates: Union[FaceDescriptor, List[FaceDescriptor], FaceDescriptorBatch]
    ) -> Union[MatchingResult, List[MatchingResult]]:
        """
        Match face descriptor vs face descriptors.

        Returns:
            List of matching results if match by several descriptors otherwise one MatchingResult.
        """
        if isinstance(candidates, FaceDescriptor):
            error = self._coreMatcher.match(reference.coreEstimation, candidates.coreEstimation)
            if error.isError:
                raise LunaSDKException(LunaVLError.fromSDKError(error))
            return error.value
        elif isinstance(candidates, FaceDescriptorBatch):
            error, matchResults = self._coreMatcher.match(reference.coreEstimation, candidates.coreEstimation)
            if error.isError:
                raise LunaSDKException(LunaVLError.fromSDKError(error))
            return matchResults
        else:
            batch = self.descriptorFactory.generateDescriptorsBatch(len(candidates))
            for candidate in candidates:
                batch.append(candidate)
            error, matchResults = self._coreMatcher.match(reference.coreEstimation, batch.coreEstimation)
            if error.isError:
                raise LunaSDKException(LunaVLError.fromSDKError(error))
            return matchResults
