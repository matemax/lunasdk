"""Module realize dynamic and dense index."""
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from lunavl.sdk.descriptors.descriptors import FaceDescriptor, FaceDescriptorBatch
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from .base import CoreIndex, IndexResult


class IndexType(Enum):
    """Available index type to save|load."""

    # dense index
    dense = "dense"
    # dynamic index
    dynamic = "dynamic"


class DynamicIndex(CoreIndex):
    """
    Dynamic Index
    """

    @property
    def descriptorsCount(self):
        """Get actual count of descriptor in internal storage."""
        return self._coreIndex.countOfIndexedDescriptors()

    def append(self, descriptor: Union[FaceDescriptor, FaceDescriptorBatch]) -> None:
        """
        Appends descriptor to internal storage.
        Args:
            descriptor: descriptor or batch of descriptors with correct length, version and data
        Raises:
            RuntimeError: if descriptor type is not supported
            LunaSDKException: if an error occurs while adding the descriptor
        """
        if isinstance(descriptor, FaceDescriptor):
            error = self._coreIndex.appendDescriptor(descriptor.coreEstimation)
        elif isinstance(descriptor, FaceDescriptorBatch):
            error = self._coreIndex.appendBatch(descriptor.coreEstimation)
        else:
            raise RuntimeError(f"Not supported descriptor class: {descriptor.__class__}")

        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

    def search(self, descriptor: FaceDescriptor, maxCount: int = 1) -> List[IndexResult]:
        """
        Search for descriptors with the shorter distance to passed descriptor.
        Args:
            descriptor: descriptor to match against index
            maxCount: max count of results (default is 1)
        Raises:
            LunaSDKException: if an error occurs while searching for descriptors
        Returns:
            list with index search results
        """
        error, resIndex = self._coreIndex.search(descriptor.coreEstimation, maxCount)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return [IndexResult(result) for result in resIndex]

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


class DenseIndex(CoreIndex):
    """
    Dense Index
    """

    def __delitem__(self, index: int):
        """Remove descriptor for a dense index is not supported."""
        raise AttributeError("'DenseIndex' object has no attribute '__delitem__'")

    def search(self, descriptor: FaceDescriptor, maxCount: Optional[int] = 1) -> List[IndexResult]:
        """
        Search for descriptors with the shorter distance to passed descriptor.
        Args:
            descriptor: descriptor to match against index
            maxCount: max count of results (default is 1)
        Raises:
            LunaSDKException: if an error occurs while searching for descriptors
        Returns:
            list with index search results
        """
        error, resIndex = self._coreIndex.search(descriptor.coreEstimation, maxCount)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return [IndexResult(result) for result in resIndex]
