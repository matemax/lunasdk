"""
Module contains core index builder.
"""
from FaceEngine import IIndexBuilderPtr

from sdk.descriptors.descriptors import BaseDescriptorBatch, BaseDescriptor
from sdk.errors.errors import LunaVLError
from sdk.errors.exceptions import LunaSDKException
from sdk.indexes.dynamic_index import DynamicIndex


class IndexBuilder:
    """
    Class index builder
    """
    __slots__ = ("_coreIndexBuilder",)

    def __init__(self, coreIndexBuilder: IIndexBuilderPtr):
        """
        Init.

        Args:
            coreIndexBuilder: index builder
        """
        self._coreIndexBuilder = coreIndexBuilder

    @property
    def indexBuilder(self):
        """
        Core index builder
        Returns:
            _indexBuilder
        """
        return self._coreIndexBuilder

    def append(self, descriptor: BaseDescriptor) -> None:
        """
        Appends descriptor to internal storage
        Args:
            descriptor: descriptor with correct length, version and data
        Raises:
            LunaSDKException: if an error occurs while adding the descriptor
        """
        error = self._coreIndexBuilder.appendDescriptor(descriptor.coreEstimation)
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
        error = self._coreIndexBuilder.appendBatch(descriptorsBatch.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

    def buildIndex(self) -> DynamicIndex:
        """
        Builds index with every descriptor appended
        Raises:
            LunaSDKException: if an error occurs while building the index
        Returns:
            DynamicIndex
        """
        error, index = self._coreIndexBuilder.buildIndex()
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return DynamicIndex(index)
