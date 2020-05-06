from typing import Optional, Union, List, Tuple, Type

from FaceEngine import IDescriptorExtractorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.estimators.body_estimators.humanwarper import HumanWarp, HumanWarpedImage
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarp, FaceWarpedImage
from lunavl.sdk.descriptors.descriptors import BaseDescriptor, BaseDescriptorFactory, BaseDescriptorBatch
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException


def estimate(
    warp: Union[HumanWarp, HumanWarpedImage, FaceWarp, FaceWarpedImage],
    descriptorFactory: BaseDescriptorFactory,
    coreEstimator: IDescriptorExtractorPtr,
    descriptor: Optional[BaseDescriptor] = None,
) -> BaseDescriptor:
    """
    Estimate face descriptor from a warp image.

    Args:
        warp: warped image
        descriptor: descriptor for saving extract result
        descriptorFactory: descriptor factory
        coreEstimator: descriptor extractor
    Returns:
        estimated descriptor
    Raises:
        LunaSDKException: if estimation failed
    """
    if descriptor is None:
        descriptor = descriptorFactory.generateDescriptor()
        coreDescriptor = descriptor.coreEstimation
    else:
        coreDescriptor = descriptor.coreEstimation

    error, optionalGS = coreEstimator.extractFromWarpedImage(warp.warpedImage.coreImage, coreDescriptor)
    if error.isError:
        raise LunaSDKException(LunaVLError.fromSDKError(error))
    descriptor.garbageScore = optionalGS
    return descriptor


def estimateDescriptorsBatch(
    warps: Union[List[Union[HumanWarp, HumanWarpedImage]], List[Union[FaceWarp, FaceWarpedImage]]],
    descriptorFactory: Type[BaseDescriptorFactory],
    coreEstimator: IDescriptorExtractorPtr,
    aggregate: bool = False,
    descriptorBatch: Optional[BaseDescriptorBatch] = None,
) -> Tuple[BaseDescriptorBatch, Union[BaseDescriptor, None]]:
    """
    Estimate a batch of descriptors from warped images.

    Args:
        warps: warped images
        aggregate:  whether to estimate  aggregate descriptor or not
        descriptorBatch: optional batch for saving descriptors
        descriptorFactory: descriptor factory
        coreEstimator: descriptor extractor
    Returns:
        tuple of batch and the aggregate descriptors (or None)
    Raises:
        LunaSDKException: if estimation failed

    """
    if descriptorBatch is None:
        descriptorBatch = descriptorFactory.generateDescriptorsBatch(len(warps))
    if aggregate:
        aggregatedDescriptor = descriptorFactory.generateDescriptor()

        error, optionalGSAggregateDescriptor, scores = coreEstimator.extractFromWarpedImageBatch(
            [warp.warpedImage.coreImage for warp in warps],
            descriptorBatch.coreEstimation,
            aggregatedDescriptor.coreEstimation,
            len(warps),
        )
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        aggregatedDescriptor.garbageScore = optionalGSAggregateDescriptor
    else:
        aggregatedDescriptor = None
        error, scores = coreEstimator.extractFromWarpedImageBatch(
            [warp.warpedImage.coreImage for warp in warps], descriptorBatch.coreEstimation, len(warps)
        )
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        descriptorBatch.scores = scores
    return descriptorBatch, aggregatedDescriptor
