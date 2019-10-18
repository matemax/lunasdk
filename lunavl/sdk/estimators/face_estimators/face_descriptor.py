"""
Module contains a face descriptor estimator

See `face descriptor`_.

"""
from typing import Union, Optional, List, Tuple

from FaceEngine import IDescriptorExtractorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, CoreExceptionWrap
from lunavl.sdk.estimators.base_estimation import BaseEstimator
from lunavl.sdk.estimators.face_estimators.warper import Warp, WarpedImage
from lunavl.sdk.faceengine.descriptors import FaceDescriptorBatch, FaceDescriptor, FaceDescriptorFactory


class FaceDescriptorEstimator(BaseEstimator):
    """
    Face descriptor estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreExtractor: IDescriptorExtractorPtr, faceDescriptorFactory: "FaceDescriptorFactory"):
        """
        Init.

        Args:
            coreExtractor: core extractor
        """
        super().__init__(coreExtractor)
        self.descriptorFactory = faceDescriptorFactory

    #  pylint: disable=W0221
    def estimate(  # type: ignore
        self, warp: Union[Warp, WarpedImage], descriptor: Optional[FaceDescriptor] = None
    ) -> FaceDescriptor:
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
            descriptor = self.descriptorFactory.generateDescriptor()
            coreDescriptor = descriptor.coreEstimation
        else:
            coreDescriptor = descriptor.coreEstimation

        optionalGS = self._coreEstimator.extractFromWarpedImage(warp.warpedImage.coreImage, coreDescriptor)
        if optionalGS.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(optionalGS))
        descriptor.garbageScore = optionalGS.value
        return descriptor

    @CoreExceptionWrap(LunaVLError.EstimationBatchDescriptorError)
    def estimateDescriptorsBatch(
        self,
        warps: List[Union[Warp, WarpedImage]],
        aggregate: bool = False,
        descriptorBatch: Optional[FaceDescriptorBatch] = None,
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
            descriptorBatch = self.descriptorFactory.generateDescriptorsBatch(len(warps))
        if aggregate:
            aggregatedDescriptor = self.descriptorFactory.generateDescriptor()

            optionalGSAggregateDescriptor, scores = self._coreEstimator.extractFromWarpedImageBatch(
                [warp.warpedImage.coreImage for warp in warps],
                descriptorBatch.coreEstimation,
                aggregatedDescriptor.coreEstimation,
                len(warps),
            )
            if optionalGSAggregateDescriptor.isError:
                raise LunaSDKException(LunaVLError.fromSDKError(optionalGSAggregateDescriptor))
            descriptorBatch.scores = scores
            aggregatedDescriptor.garbageScore = optionalGSAggregateDescriptor.value
        else:
            aggregatedDescriptor = None
            error, scores = self._coreEstimator.extractFromWarpedImageBatch(
                [warp.warpedImage.coreImage for warp in warps], descriptorBatch.coreEstimation, len(warps)
            )
            if error.isError:
                raise LunaSDKException(LunaVLError.fromSDKError(error))
            descriptorBatch.scores = scores
        return descriptorBatch, aggregatedDescriptor
