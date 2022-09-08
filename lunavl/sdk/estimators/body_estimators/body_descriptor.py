"""
Module contains a body descriptor estimator

See `body descriptor`_.

"""
from typing import List, Optional, Tuple, Union

from FaceEngine import IDescriptorExtractorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.descriptors.descriptors import BodyDescriptor, BodyDescriptorBatch, BodyDescriptorFactory

from ...async_task import AsyncTask
from ...launch_options import LaunchOptions
from ..base import BaseEstimator
from ..body_estimators.bodywarper import BodyWarp, BodyWarpedImage
from ..estimators_utils.extractor_utils import estimate, estimateDescriptorsBatch

HummanDescriptorsResult = Tuple[BodyDescriptorBatch, Union[BodyDescriptor, None]]


class BodyDescriptorEstimator(BaseEstimator):
    """
    Human body descriptor estimator.
    """

    #  pylint: disable=W0235
    def __init__(
        self,
        coreExtractor: IDescriptorExtractorPtr,
        bodyDescriptorFactory: "BodyDescriptorFactory",
        launchOptions: LaunchOptions,
    ):
        """
        Init.

        Args:
            bodyDescriptorFactory: body descriptor factory
        """
        super().__init__(coreExtractor, launchOptions)
        self.descriptorFactory = bodyDescriptorFactory

    #  pylint: disable=W0221
    def estimate(  # type: ignore
        self,
        warp: Union[BodyWarp, BodyWarpedImage],
        descriptor: Optional[BodyDescriptor] = None,
        asyncEstimate: bool = False,
    ) -> Union[BodyDescriptor, AsyncTask[BodyDescriptor]]:
        """
        Estimate body descriptor from a warp image.

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

    def estimateDescriptorsBatch(
        self,
        warps: List[Union[BodyWarp, BodyWarpedImage]],
        aggregate: bool = False,
        descriptorBatch: Optional[BodyDescriptorBatch] = None,
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
