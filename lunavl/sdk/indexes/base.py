"""
Module contains base index class and search result.
"""
from __future__ import annotations

from typing import Dict, Union

from FaceEngine import SearchResult, IIndexBuilderPtr, IDenseIndexPtr, IDynamicIndexPtr

from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.descriptors.descriptors import FaceDescriptor, FaceDescriptorBatch, FaceDescriptorFactory
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException


class IndexResult(BaseEstimation):
    """
    Class for index result
    """

    _coreEstimation: SearchResult

    def __init__(self, coreEstimation: SearchResult):
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
            float value
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


class CoreIndex:
    """
    Core index class
    """
    __slots__ = ("_coreIndex", "_descriptorFactory")

    def __init__(self, coreIndex: Union[IIndexBuilderPtr, IDenseIndexPtr, IDynamicIndexPtr],
                 descriptorFactory: FaceDescriptorFactory):
        """
        Init index.

        Args:
            coreIndex: core index class
        """
        self._coreIndex = coreIndex
        self._descriptorFactory = descriptorFactory

    @property
    def bufSize(self) -> int:
        """Get storage size with descriptors."""
        return self._coreIndex.size()

    def __getitem__(self, index: int) -> FaceDescriptor:
        """
        Get descriptor by index from internal storage.
        Args:
            index: identification of descriptors position in internal storage
        Raises:
            IndexError: if index out of range
            LunaSDKException: if an error occurs while getting descriptor
        Returns:
            descriptor
        """
        if index >= self.bufSize:
            # todo remove after fix FSDK-2897 index error
            raise IndexError(f"Descriptor index '{index}' out of range")
        descriptor = self._descriptorFactory.generateDescriptor()
        error, descriptor = self._coreIndex.descriptorByIndex(index, descriptor.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return FaceDescriptor(descriptor)

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
            # todo remove after fix FSDK-2897 index error
            raise IndexError(f"Descriptor index '{index}' out of range")
        error = self._coreIndex.removeDescriptor(index)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

    def checkDescriptorVersion(self, incomingDescriptor: Union[FaceDescriptor, FaceDescriptorBatch]) -> None:
        """
        Check descriptor version with version in index.
        Args:
            incomingDescriptor: incoming descriptor
        Raises:
            LunaSDKException: if descriptor versions do not match
        """
        descriptorVersion = incomingDescriptor.coreEstimation.getModelVersion()
        # todo: change after FSDK-2867 get descriptor version from index
        if self._descriptorFactory.descriptorVersion != descriptorVersion:
            raise LunaSDKException(
                LunaVLError.IncompatibleDescriptors.format(
                    f"mismatch of descriptor versions: expected={self._descriptorFactory.descriptorVersion} "
                    f"received={descriptorVersion}"
                )
            )