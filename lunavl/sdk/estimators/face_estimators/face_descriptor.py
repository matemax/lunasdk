"""
Module contains a face descriptor estimator

See `face descriptor`_.

"""
from typing import Union, Optional, List, Tuple, Dict, Iterator

from FaceEngine import IDescriptorExtractorPtr, IDescriptorPtr, PyIFaceEngine, \
    IDescriptorBatchPtr  # pylint: disable=E0611,E0401
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, CoreExceptionWarp
from lunavl.sdk.estimators.base_estimation import BaseEstimator, BaseEstimation
from lunavl.sdk.estimators.face_estimators.warper import Warp, WarpedImage


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
        return {"descriptor": self.coreEstimation.getData(),
                "score": self.garbageScore}

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

    def __iter__(self) -> Iterator[FaceDescriptor]:
        """
        Iterator by by batch.

        Returns:
            iterator by descriptors.
        """
        itemCount = self._coreEstimation.getMaxCount()
        for index in range(itemCount):
            yield FaceDescriptor(self._coreEstimation.getDescriptorFast(index), self.scores[index])


class FaceDescriptorEstimator(BaseEstimator):
    """
    Face descriptor estimator.
    """
    #  pylint: disable=W0235
    def __init__(self, coreExtractor: IDescriptorExtractorPtr, descriptorFactory: 'PyIFaceEngine.createDescriptor',
                 descriptorBatchFactory: 'PyIFaceEngine.createDescriptorBatch'):
        """
        Init.

        Args:
            coreExtractor: core extractor
        """
        super().__init__(coreExtractor)
        self.descriptorFactory = descriptorFactory
        self.descriptorDescriptorBatchFactory = descriptorBatchFactory

    @CoreExceptionWarp(LunaVLError.CreationDescriptorError)
    def _generateDescriptor(self) -> IDescriptorPtr:
        """
        Generate core descriptor

        Returns:
            core descriptor
        """
        return self.descriptorFactory()

    @CoreExceptionWarp(LunaVLError.CreationDescriptorError)
    def _generateDescriptorsBatch(self, size: int) -> IDescriptorBatchPtr:
        """
        Generate core descriptors batch.

        Args:
            size:batch size

        Returns:
            batch
        """
        return self.descriptorDescriptorBatchFactory(size)

    #  pylint: disable=W0221
    def estimate(self, warp: Union[Warp, WarpedImage], descriptor: Optional[FaceDescriptor] = None) -> FaceDescriptor:
        """
        Estimate face descriptor from a warp image.

        Args:
            warp: warped image
            descriptor: descriptor for saving extract result

        Returns:
            estimated descriptor
        Raises:
            LunaSDKException: if estimation failed
        """
        if descriptor is None:
            coreDescriptor = self._generateDescriptor()
        else:
            coreDescriptor = descriptor.coreEstimation

        optionalGS = self._coreEstimator.extractFromWarpedImage(warp.warpedImage.coreImage, coreDescriptor)
        if optionalGS.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(optionalGS))
        return FaceDescriptor(coreDescriptor, optionalGS.value)

    @CoreExceptionWarp(LunaVLError.EstimationBatchDescriptorError)
    def estimateDescriptorsBatch(self, warps: List[Union[Warp, WarpedImage]], aggregate: bool = False,
                                 descriptorBatch: Optional[FaceDescriptorBatch] = None
                                 ) -> Tuple[FaceDescriptorBatch, Union[FaceDescriptor, None]]:
        """
        Estimate a batch of descriptors from warped images.

        Args:
            warps: warped images
            aggregate:  whether to estimate  aggregate descriptor or not
            descriptorBatch: optional batch for saving descriptors

        Returns:
            tuple of batch and the aggregate descriptors (or None)
        Raises:
            LunaSDKException: if estimation failed

        """
        if descriptorBatch is None:
            coreDescriptorBatch = self._generateDescriptorsBatch(len(warps))
        else:
            coreDescriptorBatch = descriptorBatch.coreEstimation
        if aggregate:
            aggregatedDescriptor = self.descriptorFactory()

            optionalGSAggregateDescriptor, scores = self._coreEstimator.extractFromWarpedImageBatch(
                [warp.warpedImage.coreImage for warp in warps], coreDescriptorBatch, aggregatedDescriptor, len(warps))
            aggregatedDescriptor = FaceDescriptor(aggregatedDescriptor, optionalGSAggregateDescriptor.value)
            if optionalGSAggregateDescriptor.isError:
                raise LunaSDKException(LunaVLError.fromSDKError(optionalGSAggregateDescriptor))
        else:
            aggregatedDescriptor = None
            error, scores = self._coreEstimator.extractFromWarpedImageBatch(
                [warp.warpedImage.coreImage for warp in warps], coreDescriptorBatch, len(warps))
            if error.isError:
                raise LunaSDKException(LunaVLError.fromSDKError(error))
        return FaceDescriptorBatch(coreDescriptorBatch, scores), aggregatedDescriptor
