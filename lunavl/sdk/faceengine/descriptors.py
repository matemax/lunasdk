"""
Module contains a face descriptor estimator

See `face descriptor`_.

"""
from collections import Iterator
from typing import Dict, List
from typing import Union, Optional

from FaceEngine import IDescriptorPtr, IDescriptorBatchPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, CoreExceptionWarp
from lunavl.sdk.estimators.base_estimation import BaseEstimation


class FaceDescriptor(BaseEstimation):
    """
    Descriptor

    Attributes:
        garbageScore (float): garbage score
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: IDescriptorPtr, garbageScore: float = 0.0):
        super().__init__(coreEstimation)
        self.garbageScore = garbageScore

    def asDict(self) -> Dict[str, Union[float, bytes]]:
        """
        Convert to dict

        Returns:
            Dict with keys "descriptor" and "score"
        """
        return {"descriptor": self.coreEstimation.getData(), "score": self.garbageScore, "version": self.model}

    @property
    def rawDescriptor(self) -> bytes:
        """
        Get raw descriptors
        Returns:
            bytes with metadata
        """
        error, descBytes = self.coreEstimation.save()
        if error.isError():
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return descBytes

    @property
    def asVector(self) -> List[int]:
        """
        Convert descriptor to list of ints
        Returns:
            list of ints.
        """
        return self.coreEstimation.getDescriptor()

    @property
    def asBytes(self) -> bytes:
        """
        Get descriptor as bytes.

        Returns:

        """
        return self.coreEstimation.getData()

    @property
    def model(self) -> int:
        """
        Get model of descriptor
        Returns:
            model version
        """
        return self.coreEstimation.getModelVersion()


class FaceDescriptorBatch(BaseEstimation):
    """
    Face descriptor batch.

    Attributes:
        scores (List[float]):  list of garbage scores
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: IDescriptorBatchPtr, scores: Optional[List[float]] = None):
        super().__init__(coreEstimation)
        if scores is None:
            self.scores = [0.0 for _ in range(coreEstimation.getMaxCount())]
        else:
            self.scores = scores

    def __len__(self) -> int:
        """
        Get batch size.

        Returns:
            batch size
        """
        return self._coreEstimation.getMaxCount()

    def asDict(self) -> List[Dict]:
        """
        Get batch in json like object.

        Returns:
            list of descriptors dict
        """
        return [descriptor.asDict() for descriptor in self]

    def __getitem__(self, i) -> FaceDescriptor:
        """
        Get descriptor by index

        Args:
            i: index

        Returns:
            descriptor
        """
        return FaceDescriptor(self._coreEstimation.getDescriptorFast(i), self.scores[i])

    def __iter__(self) -> Iterator:
        """
        Iterator by by batch.

        Returns:
            iterator by descriptors.
        """
        itemCount = self._coreEstimation.getMaxCount()
        for index in range(itemCount):
            yield FaceDescriptor(self._coreEstimation.getDescriptorFast(index), self.scores[index])

    def append(self, descriptor: FaceDescriptor) -> None:
        """
        Add descriptor to end of batch.

        Args:
            descriptor: descriptor
        """
        self.coreEstimation.add(descriptor.coreEstimation)
        self.scores.append(descriptor.garbageScore)


class FaceDescriptorFactory:
    """
    Face Descriptor factory.

    Attributes:
        _faceEngine (VLFaceEngine): faceEngine
    """

    def __init__(self, faceEngine: "VLFaceEngine"):  # type: ignore # noqa: F821
        self._faceEngine = faceEngine

    @CoreExceptionWarp(LunaVLError.CreationDescriptorError)
    def generateDescriptor(self) -> IDescriptorPtr:
        """
        Generate core descriptor

        Returns:
            core descriptor
        """
        return FaceDescriptor(self._faceEngine.coreFaceEngine.createDescriptor())

    @CoreExceptionWarp(LunaVLError.CreationDescriptorError)
    def generateDescriptorsBatch(self, size: int) -> IDescriptorBatchPtr:
        """
        Generate core descriptors batch.

        Args:
            size:batch size

        Returns:
            batch
        """
        return FaceDescriptorBatch(self._faceEngine.coreFaceEngine.createDescriptorBatch(size))
