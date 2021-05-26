"""
Module contains core index builder.
"""
from pathlib import Path
from typing import Union, Type

from FaceEngine import PyIFaceEngine

from lunavl.sdk.descriptors.descriptors import (
    FaceDescriptor,
    FaceDescriptorBatch,
)
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from .base import CoreIndex
from .stored_index import DynamicIndex, IndexType, DenseIndex


class IndexBuilder(CoreIndex):
    """
    Index builder class (only supports face descriptors)

    Attributes:
        _bufSize (int): storage size with descriptors
    """

    def __init__(self, faceEngine: PyIFaceEngine):
        super().__init__(faceEngine.createIndexBuilder(), faceEngine)
        self._bufSize = 0

    @property
    def bufSize(self) -> int:
        """Get storage size with descriptors."""
        return self._bufSize

    def _getDenseIndex(self, path: str) -> DenseIndex:
        """
        Get dense index from file
        Args:
            path: path to saved index
        Raises:
            LunaSDKException: if an error occurs while loading the index
        Returns:
            dense index
        """
        error, loadedIndex = self._faceEngine.loadDenseIndex(path)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return DenseIndex(loadedIndex, self._faceEngine)

    def _getDynamicIndex(self, path: str) -> DynamicIndex:
        """
        Get dynamic index from file
        Args:
            path: path to saved index
        Raises:
            LunaSDKException: if an error occurs while loading the index
        Returns:
            dynamic index
        """
        error, loadedIndex = self._faceEngine.loadDynamicIndex(path)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return DynamicIndex(loadedIndex, self._faceEngine)

    def append(self, descriptor: FaceDescriptor) -> None:
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

    def appendBatch(self, descriptorsBatch: FaceDescriptorBatch) -> None:
        """
        Appends batch of descriptors to internal storage.
        Args:
            descriptorsBatch: batch of descriptors with correct length, version and data
        Raises:
            LunaSDKException: if an error occurs while adding the batch of descriptors
        """
        error = self._coreIndex.appendBatch(descriptorsBatch.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        self._bufSize += len(descriptorsBatch)

    def __delitem__(self, index: int) -> None:
        """
        Removes descriptor out of internal storage.
        Args:
            index: identification of descriptors position in internal storage
        Raises:
            IndexError: if index out of range
            LunaSDKException: if an error occurs while remove descriptor failed
        """
        super().__delitem__(index)
        self._bufSize -= 1

    def loadIndex(self, path: str, indexType: IndexType) -> Union[DynamicIndex, DenseIndex]:
        """
        Load 'dynamic' or 'dense' index from file.
        Args:
            path: path to saved index
            indexType: index type ('dynamic' or 'dense')
        Raises:
            FileNotFoundError: if the index file is not found
        Returns:
            class of DenseIndex or DynamicIndex
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"No such file or directory: {path}")

        if IndexType(indexType) == IndexType.dynamic:
            return self._getDynamicIndex(path=path)
        return self._getDenseIndex(path=path)

    def buildIndex(self) -> DynamicIndex:
        """
        Build index with all appended descriptors.
        Raises:
            LunaSDKException: if an error occurs while building the index
        Returns:
            DynamicIndex
        """
        error, index = self._coreIndex.buildIndex()
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return DynamicIndex(index, self._faceEngine)
