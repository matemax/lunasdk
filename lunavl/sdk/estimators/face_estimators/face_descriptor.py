"""
Module contains a face descriptor estimator

See `face descriptor`_.

"""
from typing import Union, Optional, List, Tuple

from FaceEngine import IDescriptorExtractorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.descriptors.descriptors import FaceDescriptorBatch, FaceDescriptor, FaceDescriptorFactory
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap
from ..base import BaseEstimator
from ..estimators_utils.extractor_utils import estimateDescriptorsBatch, estimate
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage


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
        self, warp: Union[FaceWarp, FaceWarpedImage], descriptor: Optional[FaceDescriptor] = None
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
        outputDescriptor = estimate(
            warp=warp,
            descriptor=descriptor,
            descriptorFactory=self.descriptorFactory,
            coreEstimator=self._coreEstimator,
        )
        return outputDescriptor  # type: ignore

    @CoreExceptionWrap(LunaVLError.EstimationBatchDescriptorError)
    def estimateDescriptorsBatch(
        self,
        warps: List[Union[FaceWarp, FaceWarpedImage]],
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
        batch, aggregatedDescriptor = estimateDescriptorsBatch(
            warps=warps,
            descriptorFactory=self.descriptorFactory,  # type: ignore
            aggregate=aggregate,
            descriptorBatch=descriptorBatch,
            coreEstimator=self._coreEstimator,
        )
        return batch, aggregatedDescriptor  # type: ignore
