from abc import abstractmethod
from typing import Any, List, Union
from FaceEngine import IDescriptorMatcherPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.estimators.face_estimators.face_descriptor import FaceDescriptor, FaceDescriptorBatch


class MatchResult:
    pass


class FaceMatcher:
    """
    Base estimator class. Class is  a container for core estimations. Mostly estimate attributes  can be get through
    a corresponding properties.

    Attributes:
        _coreMatcher (IDescriptorMatcherPtr): core matcher
    """
    __slots__ = ('_coreMatcher',)

    def __init__(self, coreMatcher: IDescriptorMatcherPtr):
        """
        Init.

        Args:
            coreMatcher: core matcher
        """
        self._coreMatcher = coreMatcher

    @abstractmethod
    def match(self, reference: FaceDescriptor,
              candidates: Union[FaceDescriptor, List[FaceDescriptor], FaceDescriptorBatch]) -> MatchResult:
        """
        Match face descriptor vs face descriptors.

        Returns:
            estimated attributes
        """
        if isinstance(candidates, FaceDescriptor):
            return self._coreMatcher.match(reference.coreEstimation, candidates.coreEstimation)
        elif isinstance(candidates, FaceDescriptorBatch):
            return self._coreMatcher.match(reference.coreEstimation, candidates.coreEstimation)
        else:
            pass
