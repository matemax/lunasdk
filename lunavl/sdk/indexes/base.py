"""
Module contains base index class and search result.
"""
from __future__ import annotations

from typing import Any, Dict, Union, Optional, List, Tuple

from FaceEngine import SearchResult, IDenseIndexPtr, IDynamicIndexPtr, FSDKErrorResult

from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.descriptors.descriptors import BaseDescriptor
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
    __slots__ = ("_coreIndex",)

    def __init__(self, coreIndex: Any):
        """
        Init index.

        Args:
            coreIndex: core index class
        """
        self._coreIndex = coreIndex

    @property
    def bufSize(self) -> int:
        """Get storage size with descriptors."""
        return self._coreIndex.size()

    def getDescriptor(self, index: int, descriptor: BaseDescriptor) -> BaseDescriptor:
        """
        Get descriptor by index from internal storage.
        Args:
            index: identification of descriptors position in internal storage
            descriptor: class container for writing the descriptor data # todo: remove descriptor after FSDK-2867
        Raises:
            IndexError: if index out of range
            LunaSDKException: if an error occurs while getting descriptor
        Returns:
            descriptor
        """
        if index >= self.bufSize:
            raise IndexError(f"Descriptor index '{index}' out of range")    # todo remove after fix FSDK index error
        error, descriptor = self._coreIndex.descriptorByIndex(index, descriptor.coreEstimation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return BaseDescriptor(descriptor)


class BaseIndex(CoreIndex):
    """
    Base class for indexes
    """

    _coreIndex: Union[IDenseIndexPtr, IDynamicIndexPtr]

    def search(self, descriptor: BaseDescriptor, maxCount: Optional[int] = 1) -> List[IndexResult]:
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

    @classmethod
    def load(cls, faceEngineResult: Tuple[FSDKErrorResult, Union[IDenseIndexPtr, IDynamicIndexPtr]]) -> BaseIndex:
        """
        Load 'dynamic' or 'dense' index from face engine result.
        Args:
            faceEngineResult: tuple with FSDKErrorResult and index
        Returns:
            class: dense or dynamic index
        """
        error, loadedIndex = faceEngineResult
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return cls(loadedIndex)
