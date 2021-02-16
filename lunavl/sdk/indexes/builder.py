"""
Module contains core index builder.
"""
from pathlib import Path
from typing import Union

from FaceEngine import PyIFaceEngine

from lunavl.sdk.descriptors.descriptors import (
    BaseDescriptorBatch,
    BaseDescriptor,
    FaceDescriptorFactory,
    HumanDescriptorFactory,
)
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from .base import CoreIndex
from .stored_index import DynamicIndex, IndexType, DenseIndex


class IndexBuilder(CoreIndex):
    """
    Index builder class

    Attributes:
        _bufSize (int): storage size with descriptors
        _faceEngine (VLFaceEngine): faceEngine
        _descriptorFactory (BaseDescriptorFactory): descriptor factory
    """

    def __init__(
            self, faceEngine: PyIFaceEngine, descriptorFactory: Union[FaceDescriptorFactory, HumanDescriptorFactory]
    ):
        super().__init__(faceEngine.createIndexBuilder(), descriptorFactory)
        self._bufSize = 0
        self._faceEngine = faceEngine
        self._descriptorFactory = descriptorFactory

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
        super().__delitem__(index=index)
        self._bufSize -= 1

    def loadIndex(self, path: str, indexType: IndexType) -> Union[DynamicIndex, DenseIndex]:
        """
        Load 'dynamic' or 'dense' index from file.
        Args:
            path: path to saved index
            indexType: index type ('dynamic' or 'dense')
        Raises:
            LunaSDKException: if an error occurs while loading the index
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"No such file or directory: {path}")
        if indexType == IndexType.dynamic:
            error, loadedIndex = self._faceEngine.loadDynamicIndex(path)
            _cls = DynamicIndex
        elif indexType == IndexType.dense:
            error, loadedIndex = self._faceEngine.loadDenseIndex(path)
            _cls = DenseIndex
        else:
            raise ValueError(f"{indexType} is not a valid, must be one of ['dynamic', 'dense']")
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return _cls(loadedIndex, self._descriptorFactory)

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
        return DynamicIndex(index, self._descriptorFactory)
