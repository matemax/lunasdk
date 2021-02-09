"""
Module contains core index builder.
"""
from FaceEngine import IIndexBuilderPtr

from lunavl.sdk.descriptors.descriptors import BaseDescriptorBatch, BaseDescriptor
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from .dynamic_index import DynamicIndex


class IndexBuilder:
    """
    Class index builder
    """
    __slots__ = ("_coreIndexBuilder", "bufSize")

    def __init__(self, coreIndexBuilder: IIndexBuilderPtr):
        """
        Init index builder.

        Args:
            coreIndexBuilder: index builder
        """
        self._coreIndexBuilder = coreIndexBuilder
        self.bufSize = 0

    @property
    def indexBuilder(self):
        """
        Core index builder.
        Returns:
            _indexBuilder
        """
        return self._coreIndexBuilder

    def append(self, descriptor: BaseDescriptor) -> None:
        """
        Appends descriptor to internal storage.
        Args:
            descriptor: descriptor with correct length, version and data
        Raises:
            LunaSDKException: if an error occurs while adding the descriptor
        """
        error = self._coreIndexBuilder.appendDescriptor(descriptor.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        self.bufSize += 1

    def appendBatch(self, descriptorsBatch: BaseDescriptorBatch) -> None:
        """
        Appends batch of descriptors to internal storage.
        Args:
            descriptorsBatch: Batch of descriptors with correct length, version and data
        Raises:
            LunaSDKException: if an error occurs while adding the batch of descriptors
        """
        error = self._coreIndexBuilder.appendBatch(descriptorsBatch.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        self.bufSize += len(descriptorsBatch)

    def getDescriptor(self, index: int, descriptor: BaseDescriptor) -> BaseDescriptor:
        """
        Get descriptor by index from internal storage.  # todo: remove descriptor after FSDK-2867
        Args:
            index: identification of descriptors position in internal storage
            descriptor: class container for writing the descriptor data
        Raises:
            IndexError: if index out of range
            LunaSDKException: if an error occurs while getting descriptor
        Returns:
            descriptor
        """
        if index >= self.bufSize:
            raise IndexError(f"Descriptor index '{index}' out of range")    # todo remove after fix FSDK index error
        error, descriptor = self._coreIndexBuilder.descriptorByIndex(index, descriptor.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return BaseDescriptor(descriptor)

    def __delitem__(self, index: int):
        """
        Removes descriptor out of internal storage.
        Args:
            index: identification of descriptors position in internal storage
        Raises:
            IndexError: if index out of range
            LunaSDKException: if an error occurs while remove descriptor failed
        """
        if index >= self.bufSize:
            raise IndexError(f"Descriptor index '{index}' out of range")    # todo remove after fix FSDK index error
        error = self._coreIndexBuilder.removeDescriptor(index)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        self.bufSize -= 1

    def buildIndex(self) -> DynamicIndex:
        """
        Builds index with every descriptor appended.
        Raises:
            LunaSDKException: if an error occurs while building the index
        Returns:
            DynamicIndex
        """
        error, index = self._coreIndexBuilder.buildIndex()
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return DynamicIndex(index)
