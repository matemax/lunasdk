from functools import partial
from typing import Union, List, Literal, overload, Tuple, Dict

from FaceEngine import (  # pylint: disable=E0611,E0401
    Image as CoreImage,
    Detection as CoreFaceDetection,
)

from lunavl.sdk.async_task import AsyncTask
from lunavl.sdk.detectors.facedetector import Landmarks5, FaceDetection, Landmarks68, FaceLandmarks
from lunavl.sdk.errors.exceptions import assertError
from ..base import BaseEstimator

PreparedBatch = List[Tuple[CoreImage, CoreFaceDetection, List[int]]]


def _prepareBatch(detections: List[FaceDetection]) -> PreparedBatch:
    """
    Prepare face detections for core estimator

    Args:
        detections: input face detections

    Returns:
        List of Tuples. Each tuple is image, Core detections on the image, detection index in original detection array.
    """
    res: Dict[int, Tuple[CoreImage, CoreFaceDetection, List[int]]] = {}
    for idx, detection in enumerate(detections):
        imageId = id(detection.image)
        imageDetections = res.get(imageId)
        coreDetection: CoreFaceDetection = detection.coreEstimation.detection
        if not imageDetections:
            res[imageId] = (detection.image.coreImage, [coreDetection], [idx])
        else:
            imageDetections[1].append(coreDetection)
            imageDetections[2].append(idx)

    return [detectionsPerImage for detectionsPerImage in res.values()]


@overload
def _postProcessingBatch(
    error,
    estimatedBatch,
    batchSize: int,
    preparedBatch: PreparedBatch,
    landmarksType: Literal[FaceLandmarks.Landmarks5],
) -> List[Landmarks5]:
    ...


@overload
def _postProcessingBatch(
    error,
    estimatedBatch,
    batchSize: int,
    preparedBatch: PreparedBatch,
    landmarksType: Literal[FaceLandmarks.Landmarks68],
) -> List[Landmarks68]:
    ...


def _postProcessingBatch(
    error,
    estimatedBatch,
    batchSize: int,
    preparedBatch: PreparedBatch,
    landmarksType: Literal[FaceLandmarks.Landmarks5, FaceLandmarks.Landmarks68],
) -> Union[List[Landmarks68], List[Landmarks5]]:
    """
    Post processing core landmarks estimations. Validate error and convert resul to float structure

    Args:
        error: estimation error
        estimatedBatch: core batch estimation
        preparedBatch: prepared batch
        landmarksType: estimated landmarks type

    Returns:
        list of landmarks
    """
    assertError(error)
    if landmarksType == FaceLandmarks.Landmarks5:
        landmarksClass = Landmarks5
        getter = estimatedBatch.getLandmarks5
        res: List[Landmarks5] = [None for _ in range(batchSize)]  # type: ignore
    else:
        landmarksClass = Landmarks68  # type: ignore
        getter = estimatedBatch.getLandmarks68
        res: List[Landmarks68] = [None for _ in range(batchSize)]  # type: ignore
    count = 0
    for imageIdx in range(len(preparedBatch)):
        allImageLandmarks = getter(imageIdx)
        detectionIndexes = preparedBatch[imageIdx][2]

        for idx, estimatedLandmarks in enumerate(allImageLandmarks):
            landmarks = landmarksClass(estimatedLandmarks)
            resultIdx = detectionIndexes[idx]
            res[resultIdx] = landmarks
            count += 1
    return res


@overload
def _postProcessing(
    error,
    estimatedBatch,
    landmarksType: Literal[FaceLandmarks.Landmarks5],
) -> Landmarks5:
    ...


@overload
def _postProcessing(
    error,
    estimatedBatch,
    landmarksType: Literal[FaceLandmarks.Landmarks68],
) -> Landmarks68:
    ...


def _postProcessing(
    error, estimatedBatch, landmarksType: Literal[FaceLandmarks.Landmarks5, FaceLandmarks.Landmarks68]
) -> Union[Landmarks5, Landmarks68]:
    """
    Post processing one core landmarks estimations. Validate error and convert result to a lunavl structure.

    Args:
        error: core error
        estimatedBatch: estimated core batch
        landmarksType: estimated landmarks type

    Returns:
        Landmarks5 or Landmarks68
    """
    assertError(error)
    landmarksClass = Landmarks5 if landmarksType == FaceLandmarks.Landmarks5 else Landmarks68
    estimatedLandmarks = estimatedBatch[0]
    landmarks = landmarksClass(estimatedLandmarks)
    return landmarks  # type: ignore


class FaceLandmarksEstimator(BaseEstimator):
    """
    Face landmarks estimator. Estimator can to predict Landmarks5  or Landmarks68
    """

    #  pylint: disable=W0221
    @overload  # type: ignore
    def estimate(
        self,
        detection: FaceDetection,
        landmarksType: Literal[FaceLandmarks.Landmarks5],
        asyncEstimate: Literal[False] = False,
    ) -> Landmarks5:
        ...

    @overload
    def estimate(
        self,
        detection: FaceDetection,
        landmarksType: Literal[FaceLandmarks.Landmarks5],
        asyncEstimate: Literal[True],
    ) -> AsyncTask[Landmarks5]:
        ...

    @overload
    def estimate(  # type: ignore
        self,
        detection: FaceDetection,
        landmarksType: Literal[FaceLandmarks.Landmarks68],
        asyncEstimate: Literal[False],
    ) -> Landmarks68:
        ...

    @overload
    def estimate(
        self,
        detection: FaceDetection,
        landmarksType: Literal[FaceLandmarks.Landmarks68],
        asyncEstimate: Literal[True],
    ) -> AsyncTask[Landmarks68]:
        ...

    def estimate(
        self,
        detection: FaceDetection,
        landmarksType: Literal[FaceLandmarks.Landmarks5, FaceLandmarks.Landmarks68],
        asyncEstimate: bool = False,
    ) -> Union[Landmarks5, Landmarks68, AsyncTask[Landmarks5], AsyncTask[Landmarks68]]:
        """
        Estimate mouth state on warp.

        Args:
            detection: face detection
            landmarksType: landmarks type for estimation
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated states if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        if landmarksType == FaceLandmarks.Landmarks5:
            postProcessing = partial(_postProcessing, landmarksType=FaceLandmarks.Landmarks5)
            estimator = (
                self._coreEstimator.asyncDetectLandmarks5 if asyncEstimate else self._coreEstimator.detectLandmarks5
            )
        else:
            postProcessing = partial(_postProcessing, landmarksType=FaceLandmarks.Landmarks68)
            estimator = (
                self._coreEstimator.asyncDetectLandmarks68 if asyncEstimate else self._coreEstimator.detectLandmarks68
            )
        if asyncEstimate:
            task = estimator(detection.image.coreImage, [detection.coreEstimation.detection])
            return AsyncTask(task, postProcessing)
        error, landmarks = estimator(detection.image.coreImage, [detection.coreEstimation.detection])
        return postProcessing(error, landmarks)

    @overload
    def estimateBatch(
        self,
        detections: list[FaceDetection],
        landmarksType: Literal[FaceLandmarks.Landmarks5],
        asyncEstimate: Literal[False],
    ) -> List[Landmarks5]:
        ...

    @overload
    def estimateBatch(
        self,
        detections: list[FaceDetection],
        landmarksType: Literal[FaceLandmarks.Landmarks5],
        asyncEstimate: Literal[True],
    ) -> AsyncTask[List[Landmarks5]]:
        ...

    @overload
    def estimateBatch(
        self,
        detections: list[FaceDetection],
        landmarksType: Literal[FaceLandmarks.Landmarks68],
        asyncEstimate: Literal[False],
    ) -> List[Landmarks68]:
        ...

    @overload
    def estimateBatch(
        self,
        detections: list[FaceDetection],
        landmarksType: Literal[FaceLandmarks.Landmarks68],
        asyncEstimate: Literal[True],
    ) -> AsyncTask[List[Landmarks68]]:
        ...

    def estimateBatch(
        self,
        detections: list[FaceDetection],
        landmarksType: Literal[FaceLandmarks.Landmarks5, FaceLandmarks.Landmarks68],
        asyncEstimate: bool = False,
    ) -> Union[List[Landmarks5], List[Landmarks68], AsyncTask[List[Landmarks68]], AsyncTask[List[Landmarks5]]]:
        """
        Estimate mouth state on warp.

        Args:
            detections: face detection
            landmarksType: landmarks type for estimation
            asyncEstimate: estimate or run estimation in background

        Returns:
            estimated states if asyncEstimate is False otherwise async task
        Raises:
            LunaSDKException: if estimation failed
        """
        preparedBatch = _prepareBatch(detections)
        coreImages = [image[0] for image in preparedBatch]
        coreDetections = [image[1] for image in preparedBatch]
        batchSize = len(detections)
        if landmarksType == FaceLandmarks.Landmarks5:
            postProcessing = partial(
                _postProcessingBatch,
                preparedBatch=preparedBatch,
                batchSize=batchSize,
                landmarksType=FaceLandmarks.Landmarks5,
            )
            estimator = (
                self._coreEstimator.asyncDetectLandmarks5 if asyncEstimate else self._coreEstimator.detectLandmarks5
            )
        else:
            postProcessing = partial(
                _postProcessingBatch,
                preparedBatch=preparedBatch,
                batchSize=batchSize,
                landmarksType=FaceLandmarks.Landmarks68,
            )
            estimator = (
                self._coreEstimator.asyncDetectLandmarks68 if asyncEstimate else self._coreEstimator.detectLandmarks68
            )

        if asyncEstimate:
            task = estimator(coreImages, coreDetections)
            return AsyncTask(task, postProcessing)
        error, estimations = estimator(coreImages, coreDetections)
        return postProcessing(error, estimations, preparedBatch=preparedBatch, batchSize=batchSize)
