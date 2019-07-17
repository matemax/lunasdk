from lunavl.sdk.estimators.base_estimation import BaseEstimator, BaseEstimation
from FaceEngine import IDescriptorExtractorPtr, IDescriptorPtr, PyIFaceEngine, \
    IDescriptorBatchPtr  # pylint: disable=E0611,E0401
from typing import Union, Optional, List

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

    def asDict(self) -> Union[dict, list]:
        return {"descriptor": self.coreEstimation.getData()}

    def asDict(self) -> Union[dict, list]:
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
        return {"descriptors": [descriptor.asDict() for descriptor in self]}

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
        res = self._coreEstimator.extractFromWarpedImage(warp.warpedImage.coreImage, descriptor)
        if res.isError:
            raise ValueError("12343")
        return FaceDescriptor(descriptor, res.value)

    def estimateWarpsBatch(self, warps: List[Union[Warp, WarpedImage]], aggregate: bool = False,
                           descriptorBatch: Optional[FaceDescriptorBatch] = None):
        if descriptorBatch is not None:
            # if (len(warps) != len(FaceDescriptorBatch)) and (aggregate and len(FaceDescriptorBatch) == 1):
            #     raise ValueError("12343")
            pass
        else:
            descriptorBatch = self.descriptorDescriptorBatchFactory(len(warps))
        if aggregate:
            res, scores = self._coreEstimator.extractFromWarpedImageBatch(
                [warp.warpedImage.coreImage for warp in warps],
                descriptorBatch, aggregate, len(warps))
        else:
            res, scores = self._coreEstimator.extractFromWarpedImageBatch(
                [warp.warpedImage.coreImage for warp in warps],
                descriptorBatch, len(warps))
        if res.isError:
            raise ValueError("12343")
        return FaceDescriptorBatch(descriptorBatch, scores)
