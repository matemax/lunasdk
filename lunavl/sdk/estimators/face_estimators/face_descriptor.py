from lunavl.sdk.estimators.base_estimation import BaseEstimator, BaseEstimation
from FaceEngine import IDescriptorExtractorPtr, IDescriptorPtr, PyIFaceEngine, \
    IDescriptorBatchPtr  # pylint: disable=E0611,E0401
from typing import Union, Optional, List

from lunavl.sdk.estimators.face_estimators.warper import Warp, WarpedImage


class FaceDescriptorBatch(BaseEstimation):

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: IDescriptorBatchPtr):
        super().__init__(coreEstimation)

    def asDict(self) -> Union[dict, list]:
        return {"descriptor": self.coreEstimation.getData()}


class FaceDescriptor(BaseEstimation):
    """
    Descriptor
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: IDescriptorPtr):
        super().__init__(coreEstimation)

    def asDict(self) -> Union[dict, list]:
        return {"descriptor": self.coreEstimation.getData()}

    def asDict(self) -> Union[dict, list]:
        return {"descriptor": self.coreEstimation.getData()}

    @property
    def rawDescriptor(self):
        return self.coreEstimation.getData()

    @property
    def prettyDescriptor(self):
        return self.coreEstimation.getDescriptor()

    @property
    def model(self):
        return self.coreEstimation.getModelVersion()


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
        return FaceDescriptor(descriptor)

    def estimateWarpsBatch(self, warps: List[Union[Warp, WarpedImage]], aggregate: bool = False,
                           descriptorBatch: Optional[FaceDescriptorBatch] = None):
        if descriptorBatch is not None:
            # if (len(warps) != len(FaceDescriptorBatch)) and (aggregate and len(FaceDescriptorBatch) == 1):
            #     raise ValueError("12343")
            pass
        else:
            descriptorBatch = self.descriptorDescriptorBatchFactory(len(warps))
        if aggregate:
            self._coreEstimator.extractFromWarpedImageBatch([warp.warpedImage.coreImage for warp in warps],
                                                            descriptorBatch, aggregate, len(warps))
        else:
            self._coreEstimator.extractFromWarpedImageBatch([warp.warpedImage.coreImage for warp in warps],
                                                            descriptorBatch, len(warps))
        return descriptorBatch

