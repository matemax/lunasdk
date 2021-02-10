"""
Module contains core index builder.
"""
from FaceEngine import IIndexBuilderPtr

from lunavl.sdk.descriptors.descriptors import BaseDescriptorBatch, BaseDescriptor
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from .base import CoreIndex
from .stored_index import DynamicIndex


class IndexBuilder(CoreIndex):
    """
    Index builder class
    """

    _coreIndex: IIndexBuilderPtr

    def __init__(self, indexBuilder: IIndexBuilderPtr):
        """
        Init index builder.
        """
        super().__init__(coreIndex=indexBuilder)
        self._bufSize = 0

    @property
    def bufSize(self) -> int:
        """Get storage size with descriptors."""
        return self._bufSize

    def append(self, descriptor: BaseDescriptor) -> None:
        """
        Appends descriptor to internal storage.
        Args:
            descriptor: descriptor with correct length, version and data
        Raises:
            LunaSDKException: if an error occurs while adding the descriptor
        """
        error = self._coreIndex.appendDescriptor(descriptor.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        self._bufSize += 1

    def appendBatch(self, descriptorsBatch: BaseDescriptorBatch) -> None:
        """
        Appends batch of descriptors to internal storage.
        Args:
            descriptorsBatch: Batch of descriptors with correct length, version and data
        Raises:
            LunaSDKException: if an error occurs while adding the batch of descriptors
        """
        error = self._coreIndex.appendBatch(descriptorsBatch.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        self._bufSize += len(descriptorsBatch)

    def __delitem__(self, index: int):
        """
        Removes descriptor out of internal storage.
        Args:
            index: identification of descriptors position in internal storage
        Raises:
            IndexError: if index out of range
            LunaSDKException: if an error occurs while remove descriptor failed
        """
        if index >= self._bufSize:
            raise IndexError(f"Descriptor index '{index}' out of range")    # todo remove after fix FSDK index error
        error = self._coreIndex.removeDescriptor(index)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        self._bufSize -= 1

    def buildIndex(self) -> DynamicIndex:
        """
        Builds index with every descriptor appended.
        Raises:
            LunaSDKException: if an error occurs while building the index
        Returns:
            DynamicIndex
        """
        error, index = self._coreIndex.buildIndex()
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return DynamicIndex(index)
