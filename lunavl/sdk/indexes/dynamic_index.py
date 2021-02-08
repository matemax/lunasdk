from typing import List, Union, Dict

from FaceEngine import SearchResult as _SearchResult, IDynamicIndexPtr

from sdk.base import BaseEstimation
from sdk.descriptors.descriptors import BaseDescriptor, BaseDescriptorBatch
from sdk.errors.errors import LunaVLError
from sdk.errors.exceptions import LunaSDKException


class IndexResult(BaseEstimation):
    """
    Class for index result
    """

    _coreEstimation: _SearchResult

    def __init__(self, coreEstimation: _SearchResult):
        """
        Init index result.
        Args:
            coreEstimation: core index result
        """
        super().__init__(coreEstimation)

    @property
    def distance(self) -> float:
        """
        Get descriptor distance
        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.distance

    @property
    def similarity(self) -> float:
        """
        Get descriptor similarity
        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.similarity

    @property
    def index(self) -> int:
        """
        Get descriptor index
        Returns:
            int value
        """
        return self._coreEstimation.index

    def asDict(self) -> Dict[str, Union[float, int]]:
        """
        Convert index search result to dict
        Returns:
            dict of index results
        """
        return {"distance": self.distance, "similarity": self.similarity, "index": self.index}


class DynamicIndex:
    """
    Dynamic Index
    """

    __slots__ = ("_coreDynamicIndex",)

    def __init__(self, coreDynamicIndex: IDynamicIndexPtr):
        self._coreDynamicIndex = coreDynamicIndex

    @property
    def size(self) -> int:
        """Get storage size with indexes."""
        return self._coreDynamicIndex.size()

    def append(self, descriptor: BaseDescriptor) -> None:
        """
        Appends descriptor to internal storage
        Args:
            descriptor: descriptor with correct length, version and data
        Raises:
            LunaSDKException: if an error occurs while adding the descriptor
        """
        error = self._coreDynamicIndex.appendDescriptor(descriptor.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

    def appendBatch(self, descriptorsBatch: BaseDescriptorBatch) -> None:
        """
        Appends batch of descriptors to internal storage
        Args:
            descriptorsBatch: Batch of descriptors with correct length, version and data
        Raises:
            LunaSDKException: if an error occurs while adding the batch of descriptors
        """
        error = self._coreDynamicIndex.appendBatch(descriptorsBatch.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

    def search(self, descriptor: BaseDescriptor, maxCount: int) -> List[IndexResult]:
        """
        Search for descriptors with the shorter distance to passed descriptor
        Args:
            descriptor: descriptor to match against index
            maxCount: max count of results
        Raises:
            LunaSDKException: if an error occurs while searching for descriptors
        Returns:
            list with index search results
        """
        error, resIndex = self._coreDynamicIndex.search(descriptor.coreEstimation, maxCount)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return [IndexResult(result) for result in resIndex]
