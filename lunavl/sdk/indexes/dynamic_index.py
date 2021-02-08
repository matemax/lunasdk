from typing import List

from FaceEngine import SearchResult, IDynamicIndexPtr

from sdk.descriptors.descriptors import BaseDescriptor, BaseDescriptorBatch
from sdk.errors.errors import LunaVLError
from sdk.errors.exceptions import LunaSDKException


class DynamicIndex:
    """
    Dynamic Index
    """

    __slots__ = ["_dynamicIndex"]

    def __init__(self, dynamicIndex: IDynamicIndexPtr):
        self._dynamicIndex = dynamicIndex

    @property
    def size(self) -> int:
        """Get storage size with indexes."""
        return self._dynamicIndex.size()

    def append(self, descriptor: BaseDescriptor) -> None:
        """
        Appends descriptor to internal storage
        Args:
            descriptor: descriptor with correct length, version and data
        Raises:
            LunaSDKException: if an error occurs while adding the descriptor
        """
        error = self._dynamicIndex.appendDescriptor(descriptor.coreEstimation)
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
        error = self._dynamicIndex.appendBatch(descriptorsBatch.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

    def search(self, descriptor: BaseDescriptor, maxCount: int) -> List[SearchResult]:
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
        error, result = self._dynamicIndex.search(descriptor.coreEstimation, maxCount)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return result
