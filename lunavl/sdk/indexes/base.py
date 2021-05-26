"""
Module contains base index class and search result.
"""
from __future__ import annotations

from typing import Dict, Union

from FaceEngine import SearchResult, IIndexBuilderPtr, IDenseIndexPtr, IDynamicIndexPtr, PyIFaceEngine

from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.descriptors.descriptors import FaceDescriptor
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

    __slots__ = ("_coreIndex", "_faceEngine")

    def __init__(self, coreIndex: Union[IIndexBuilderPtr, IDenseIndexPtr, IDynamicIndexPtr], faceEngine: PyIFaceEngine):
        """
        Init index.

        Args:
            coreIndex: core index class
            faceEngine (PyIFaceEngine): core face engine
        """
        self._coreIndex = coreIndex
        self._faceEngine = faceEngine

    @property
    def bufSize(self) -> int:
        """Get storage size with descriptors."""
        return self._coreIndex.size()

    @property
    def descriptorVersion(self) -> int:
        """Get descriptor version from index."""
        return self._coreIndex.getDescriptorVersion()

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
        _coreDescriptor = self._faceEngine.createDescriptor(self._coreIndex.getDescriptorVersion())
        error, descriptor = self._coreIndex.descriptorByIndex(index, _coreDescriptor)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return FaceDescriptor(descriptor)

    def __delitem__(self, index: int) -> None:
        """
        Descriptor will be removed from the graph (not from the internal storage), so it is not available for search.
        Args:
            index: identification of descriptors position in internal storage
        Raises:
            IndexError: if index out of range
            LunaSDKException: if an error occurs while remove descriptor failed
        """
        error = self._coreIndex.removeDescriptor(index)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
