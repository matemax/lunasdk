"""Common descriptor extractor utils"""
from functools import partial
from typing import Optional, Union, List, Tuple, Type

from FaceEngine import IDescriptorExtractorPtr, FSDKError  # pylint: disable=E0611,E0401

from lunavl.sdk.descriptors.descriptors import BaseDescriptor, BaseDescriptorFactory, BaseDescriptorBatch
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, assertError
from lunavl.sdk.async_task import AsyncTask
from ..body_estimators.humanwarper import HumanWarp, HumanWarpedImage
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage


def postProcessing(error, gs, descriptor):
    assertError(error)
    descriptor.garbageScore = gs
    return descriptor


def postProcessingBatch(error, gScores, descriptorBatch):
    assertError(error)
    descriptorBatch.scores = gScores
    return descriptorBatch, None


def postProcessingBatchWithAggregation(error, aggregetionGs, gScores, descriptorBatch, aggregatedDescriptor):
    assertError(error)
    aggregatedDescriptor.garbageScore = aggregetionGs
    descriptorBatch.scores = gScores
    return descriptorBatch, aggregatedDescriptor


def estimate(
    warp: Union[HumanWarp, HumanWarpedImage, FaceWarp, FaceWarpedImage],
    descriptorFactory: BaseDescriptorFactory,
    coreEstimator: IDescriptorExtractorPtr,
    descriptor: Optional[BaseDescriptor] = None,
    asyncEstimate: bool = False,
) -> Union[BaseDescriptor, AsyncTask[BaseDescriptor]]:
    """
    Estimate a face descriptor or a human descriptor from the warped image.

    Args:
        warp: warped image
        descriptor: descriptor for saving extract result
        descriptorFactory: descriptor factory
        coreEstimator: descriptor extractor
        asyncEstimate: estimate or run estimation in background
    Returns:
        estimated descriptor if asyncEstimate is false otherwise async task
    Raises:
        LunaSDKException: if estimation failed
    """
    if descriptor is None:
        descriptor = descriptorFactory.generateDescriptor()
        coreDescriptor = descriptor.coreEstimation
    else:
        coreDescriptor = descriptor.coreEstimation
    if asyncEstimate:
        task = coreEstimator.asyncExtractFromWarpedImage(warp.warpedImage.coreImage, coreDescriptor)
        return AsyncTask(task, partial(postProcessing, descriptor=descriptor))
    error, optionalGS = coreEstimator.extractFromWarpedImage(warp.warpedImage.coreImage, coreDescriptor)
    return postProcessing(error, optionalGS, descriptor)


def validateInputByBatchEstimator(estimator, *args):
    """
    Validate input data using batch estimator

    Args:
        estimator: estimator
        args: args to validate by estimator

    Raises:
         LunaSDKException(LunaVLError.BatchedInternalError): if data is not valid
         LunaSDKException: if validation is failed
    """
    validationError, sdkErrors = estimator.validate(*args)
    if validationError.isOk:
        return
    errors = [LunaVLError.fromSDKError(error) for error in sdkErrors]

    if validationError.error != FSDKError.ValidationFailed:
        raise LunaSDKException(LunaVLError.fromSDKError(validationError), errors)

    raise LunaSDKException(
        LunaVLError.BatchedInternalError.format(LunaVLError.fromSDKError(validationError).detail), errors
    )


def estimateDescriptorsBatch(
    warps: Union[List[Union[HumanWarp, HumanWarpedImage]], List[Union[FaceWarp, FaceWarpedImage]]],
    descriptorFactory: Type[BaseDescriptorFactory],
    coreEstimator: IDescriptorExtractorPtr,
    aggregate: bool = False,
    descriptorBatch: Optional[BaseDescriptorBatch] = None,
    asyncEstimate: bool = False,
) -> Union[
    Tuple[BaseDescriptorBatch, Union[BaseDescriptor, None]],
    AsyncTask[Tuple[BaseDescriptorBatch, Union[BaseDescriptor, None]]],
]:
    """
    Estimate a batch of descriptors from warped images.

    Args:
        warps: warped images
        aggregate:  whether to estimate  aggregate descriptor or not
        descriptorBatch: optional batch for saving descriptors
        descriptorFactory: descriptor factory
        coreEstimator: descriptor extractor
        asyncEstimate: estimate or run estimation in background
    Returns:
        tuple of batch and the aggregate descriptors (or None) if asyncEstimate is false otherwise async task
    Raises:
        LunaSDKException: if estimation failed
    """

    if descriptorBatch is None:
        descriptorBatch = descriptorFactory.generateDescriptorsBatch(len(warps))
    coreImages = [warp.warpedImage.coreImage for warp in warps]
    validateInputByBatchEstimator(coreEstimator, coreImages)
    if aggregate:
        aggregatedDescriptor = descriptorFactory.generateDescriptor()
        if asyncEstimate:
            task = coreEstimator.asyncExtractFromWarpedImageBatch(
                coreImages, descriptorBatch.coreEstimation, aggregatedDescriptor.coreEstimation
            )
            return AsyncTask(
                task,
                postProcessing=partial(
                    postProcessingBatchWithAggregation,
                    descriptorBatch=descriptorBatch,
                    aggregatedDescriptor=aggregatedDescriptor,
                ),
            )
        error, optionalGSAggregateDescriptor, scores = coreEstimator.extractFromWarpedImageBatch(
            coreImages, descriptorBatch.coreEstimation, aggregatedDescriptor.coreEstimation
        )
        return postProcessingBatchWithAggregation(
            error,
            optionalGSAggregateDescriptor,
            scores,
            descriptorBatch=descriptorBatch,
            aggregatedDescriptor=aggregatedDescriptor,
        )
    if asyncEstimate:
        task = coreEstimator.asyncExtractFromWarpedImageBatch(coreImages, descriptorBatch.coreEstimation)
        return AsyncTask(task, postProcessing=partial(postProcessingBatch, descriptorBatch=descriptorBatch))
    error, scores = coreEstimator.extractFromWarpedImageBatch(coreImages, descriptorBatch.coreEstimation)
    return postProcessingBatch(error, scores, descriptorBatch=descriptorBatch)
