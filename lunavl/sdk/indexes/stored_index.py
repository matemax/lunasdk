"""Module realize dynamic and dense index."""
from enum import Enum

from pathlib import Path
from FaceEngine import IDynamicIndexPtr, IDenseIndexPtr

from lunavl.sdk.descriptors.descriptors import BaseDescriptor, BaseDescriptorBatch
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from .base import BaseIndex


class IndexType(Enum):
    """Available index type to save."""

    # dense index
    dense = "dense"
    # dynamic index
    dynamic = "dynamic"


class DynamicIndex(BaseIndex):
    """
    Dynamic Index
    """

    _coreIndex: IDynamicIndexPtr

    @property
    def count(self):
        """Get actual count of descriptor in internal storage."""
        return self._coreIndex.countOfIndexedDescriptors()

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

    def __delitem__(self, index: int):
        """
        Descriptor will be removed from the graph (not from the internal storage), so it is not available for search.
        Args:
            index: identification of descriptors position in internal storage
        Raises:
            IndexError: if index out of range
            LunaSDKException: if an error occurs while remove descriptor failed
        """
        if index >= self.bufSize:
            raise IndexError(f"Descriptor index '{index}' out of range")    # todo remove after fix FSDK index error
        error = self._coreIndex.removeDescriptor(index)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

    def save(self, path: str, indexType: IndexType):
        """
        Save index as 'dynamic' or 'dense' to local storage.
        Args:
            path: path to file to be created
            indexType: index type ('dynamic' or 'dense')
        Raises:
            LunaSDKException: if an error occurs while saving the index
        """
        if Path(path).is_dir():
            raise ValueError(f"{path} must not be a directory")
        if indexType == IndexType.dynamic:
            error = self._coreIndex.saveToDynamicIndex(path)
        elif indexType == IndexType.dense:
            error = self._coreIndex.saveToDenseIndex(path)
        else:
            raise ValueError(f"{indexType} is not a valid, must be one of ['dynamic', 'dense']")
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))


class DenseIndex(BaseIndex):
    """
    Dense Index (read only)
    """

    _coreIndex: IDenseIndexPtr
