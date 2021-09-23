"""
Module contains a human descriptor estimator

See `human descriptor`_.

"""
from typing import Union, Optional, List, Tuple

from FaceEngine import IDescriptorExtractorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.descriptors.descriptors import HumanDescriptorBatch, HumanDescriptor, HumanDescriptorFactory
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap
from ..base import BaseEstimator
from ..body_estimators.humanwarper import HumanWarp, HumanWarpedImage
from ..estimators_utils.extractor_utils import estimateDescriptorsBatch, estimate
from ...async_task import AsyncTask

HummanDescriptorsResult = Tuple[HumanDescriptorBatch, Union[HumanDescriptor, None]]


class HumanDescriptorEstimator(BaseEstimator):
    """
    Human descriptor estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreExtractor: IDescriptorExtractorPtr, humanDescriptorFactory: "HumanDescriptorFactory"):
        """
        Init.

        Args:
            coreExtractor: core extractor
        """
        super().__init__(coreExtractor)
        self.descriptorFactory = humanDescriptorFactory

    #  pylint: disable=W0221
    def estimate(  # type: ignore
        self,
        warp: Union[HumanWarp, HumanWarpedImage],
        descriptor: Optional[HumanDescriptor] = None,
        asyncEstimate: bool = False,
    ) -> Union[HumanDescriptor, AsyncTask[HumanDescriptor]]:
        """
        Estimate human descriptor from a warp image.

        Args:
            warp: warped image
            descriptor: descriptor for saving extract result
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated descriptor
        Raises:
            LunaSDKException: if estimation failed
        """
        return estimate(  # type: ignore
            warp=warp,
            descriptor=descriptor,
            descriptorFactory=self.descriptorFactory,
            coreEstimator=self._coreEstimator,
            asyncEstimate=asyncEstimate,
        )

    @CoreExceptionWrap(LunaVLError.EstimationBatchDescriptorError)
    def estimateDescriptorsBatch(
        self,
        warps: List[Union[HumanWarp, HumanWarpedImage]],
        aggregate: bool = False,
        descriptorBatch: Optional[HumanDescriptorBatch] = None,
        asyncEstimate: bool = False,
    ) -> Union[HummanDescriptorsResult, AsyncTask[HummanDescriptorsResult]]:
        """
        Estimate a batch of descriptors from warped images.

        Args:
            warps: warped images
            aggregate: whether to estimate an aggregated descriptor or not
            descriptorBatch: optional batch for saving descriptors
            asyncEstimate: estimate or run estimation in background

        Returns:
            tuple with a batch and a aggregated descriptor (or None)
        Raises:
            LunaSDKException: if estimation failed

        """
        return estimateDescriptorsBatch(  # type: ignore
            warps=warps,
            descriptorFactory=self.descriptorFactory,  # type: ignore
            aggregate=aggregate,
            descriptorBatch=descriptorBatch,
            coreEstimator=self._coreEstimator,
            asyncEstimate=asyncEstimate,
        )
