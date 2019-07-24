from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, CoreExceptionWarp
from lunavl.sdk.estimators.base_estimation import BaseEstimator, BaseEstimation
from FaceEngine import IDescriptorExtractorPtr, IDescriptorPtr, PyIFaceEngine, \
    IDescriptorBatchPtr  # pylint: disable=E0611,E0401
from typing import Union, Optional, List, Tuple

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

    def asDict(self) -> dict:
        return {"descriptor": self.coreEstimation.getData(),
                "score": self.garbageScore}

    @property
    def rawDescriptor(self):
        err, bytes = self.coreEstimation.save()
        if err.isError():
            raise ValueError(234567)
        return bytes

    @property
    def prettyDescriptor(self):
        return self.coreEstimation.getDescriptor()

    @property
    def asBytes(self):
        return self.coreEstimation.getData()

    @property
    def model(self):
        return self.coreEstimation.getModelVersion()


class FaceDescriptorBatch(BaseEstimation):

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: IDescriptorBatchPtr, scores: Optional[List[float]] = None):
        super().__init__(coreEstimation)
        if scores is None:
            self.scores = [0.0 for _ in range(coreEstimation.getMaxCount())]
        else:
            self.scores = scores

    def __len__(self):
        return self._coreEstimation.getMaxCount()

    def asDict(self) -> Union[dict, list]:
        return [descriptor.asDict() for descriptor in self]

    def __getitem__(self, i):
        return FaceDescriptor(self._coreEstimation.getDescriptorFast(i), self.scores[i])

    def __iter__(self) -> FaceDescriptor:
        itemCount = self._coreEstimation.getMaxCount()
        for index in range(itemCount):
            yield FaceDescriptor(self._coreEstimation.getDescriptorFast(index), self.scores[index])


class FaceDescriptorEstimator(BaseEstimator):

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
    def genereteDescriptor(self):
        return self.descriptorFactory()

    @CoreExceptionWarp(LunaVLError.CreationDescriptorError)
    def generateDescriptorsBatch(self, size: int):
        return self.descriptorDescriptorBatchFactory(size)

    #  pylint: disable=W0221
    def estimate(self, warp: Union[Warp, WarpedImage], descriptor: Optional[FaceDescriptor] = None) -> FaceDescriptor:
        """
        Estimate emotion on warp.

        Args:
            warp: warped image
            descriptor: descriptor for saving extract result

        Returns:
            estimated descriptor
        """
        if descriptor is None:
            descriptor = self.descriptorFactory()
        optionalGS = self._coreEstimator.extractFromWarpedImage(warp.warpedImage.coreImage, descriptor)
        if optionalGS.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(optionalGS))
        return FaceDescriptor(descriptor, optionalGS.value)

    @CoreExceptionWarp(LunaVLError.EstimationBatchDescriptorError)
    def estimateWarpsBatch(self, warps: List[Union[Warp, WarpedImage]], aggregate: bool = False,
                           descriptorBatch: Optional[FaceDescriptorBatch] = None) -> Tuple[FaceDescriptorBatch,
                                                                                           FaceDescriptor]:
        if descriptorBatch is not None:
            pass
        else:
            descriptorBatch = self.generateDescriptorsBatch(len(warps))
        if aggregate:
            aggregatedDescriptor = self.descriptorFactory()

            optionalGSAggregateDescriptor, scores = self._coreEstimator.extractFromWarpedImageBatch(
                [warp.warpedImage.coreImage for warp in warps], descriptorBatch, aggregatedDescriptor, len(warps))
            aggregatedDescriptor = FaceDescriptor(aggregatedDescriptor, optionalGSAggregateDescriptor.value)
            if optionalGSAggregateDescriptor.isError:
                raise LunaSDKException(LunaVLError.fromSDKError(optionalGSAggregateDescriptor))
        else:
            aggregatedDescriptor = None
            error, scores = self._coreEstimator.extractFromWarpedImageBatch(
                [warp.warpedImage.coreImage for warp in warps], descriptorBatch, len(warps))
            if error.isError:
                raise LunaSDKException(LunaVLError.fromSDKError(error))
        return FaceDescriptorBatch(descriptorBatch, scores), aggregatedDescriptor
