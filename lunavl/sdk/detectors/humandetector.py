"""
Module contains function for detection on images.
"""
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union, overload

from FaceEngine import (  # pylint: disable=E0611,E0401
    Face,
    Human,
    FSDKErrorResult,
)

from .bodydetector import BodyDetection
from .facedetector import FaceDetection
from ..async_task import AsyncTask
from ..detectors.base import (
    ImageForDetection,
    getArgsForCoreDetectorForImages,
    validateBatchDetectInput,
)
from ..errors.exceptions import assertError
from ..image_utils.image import VLImage
from ..launch_options import LaunchOptions


class HumanDetection:
    """
    Human detection is union of face and body detection of one human.Face and body may be None if detector did not
    detect corresponding part of the torso

    Attributes:
        face (Optional[FaceDetection]): optional body detection
        body (Optional[BodyDetection]): optional face detection
        associationScore (Optional[float]): body and face association score
    """

    __slots__ = ("face", "body", "image", "associationScore")

    def __init__(
        self,
        coreFaceDetection: Optional[Face],
        coreBodyDetection: Optional[Human],
        image: VLImage,
        score: Optional[float] = None,
    ):
        """
        Init.

        Args:
            coreFaceDetection: core face detection
            coreBodyDetection: core body detection
            image: original image
            score:  association score of body and face. If face or body is none score must be set to None
        """
        self.face = FaceDetection(coreFaceDetection, image) if coreFaceDetection else None
        self.body = BodyDetection(coreBodyDetection, image) if coreBodyDetection else None
        self.associationScore = score
        self.image = image

    def asDict(self) -> Dict[str, Any]:
        """
        Convert human detection to dict (json).

        Returns:
            dict. required keys: 'rect', 'score'. optional keys: 'landmarks5', 'landmarks68'
        """
        res = {
            "face": self.face.asDict() if self.face else None,
            "body": self.body.asDict() if self.body else None,
            "association_score": self.associationScore,
        }
        return res


# alias for detection result
HumanDetectResult = List[List[HumanDetection]]


def collectDetectionsResult(
    fsdkDetectRes,
    images: List[Union[VLImage, ImageForDetection]],
) -> HumanDetectResult:
    """
    Collect detection results from core reply and prepare human detections
    Args:
        fsdkDetectRes: fsdk (re)detect results
        images: incoming images
    Returns:
        return list of lists detection, order of detection lists is corresponding to order input images
    Raises:
        RuntimeError: if any detection is not valid and it is not redection result
    """
    res = []
    for imageIdx in range(fsdkDetectRes.getSize()):
        image = images[imageIdx]
        vlImage = image if isinstance(image, VLImage) else image.image

        humanDetections = []
        bodies = []
        for bodyDetection in fsdkDetectRes.getHumanDetections(imageIdx):
            coreBody = Human()
            coreBody.img = vlImage.coreImage
            coreBody.detection = bodyDetection
            bodies.append(coreBody)
        faces = []
        for faceDetection in fsdkDetectRes.getFaceDetections(imageIdx):
            coreFace = Face()
            coreFace.img = vlImage.coreImage
            coreFace.detection = faceDetection
            faces.append(coreFace)

        associations = fsdkDetectRes.getAssociations(imageIdx)
        facesWithBody = set()
        bodiesWithFace = set()

        for association in associations:
            faceIdx = association.faceId
            bodyIdx = association.humanId
            facesWithBody.add(faceIdx)
            bodiesWithFace.add(bodyIdx)
            humanDetection = HumanDetection(faces[faceIdx], bodies[bodyIdx], score=association.score, image=vlImage)
            humanDetections.append(humanDetection)

        for bodyIdx, body in enumerate(bodies):
            if bodyIdx in bodiesWithFace:
                continue
            humanDetection = HumanDetection(None, bodies[bodyIdx], image=vlImage)
            humanDetections.append(humanDetection)

        for faceIdx, face in enumerate(faces):
            if faceIdx in bodiesWithFace:
                continue
            humanDetection = HumanDetection(faces[faceIdx], None, image=vlImage)
            humanDetections.append(humanDetection)

        res.append(humanDetections)

    return res


def postProcessing(
    error: FSDKErrorResult, detectionsBatch, images: List[Union[VLImage, ImageForDetection]]
) -> List[List[HumanDetection]]:
    """
    Convert core hbodies and faces detections from detector results to `HumanDetection` and error check.

    Args:
        error: detection error, usually error.isError is False
        detectionsBatch: core detection batch
        images: original images

    Returns:
        list, each item is face detections on corresponding image
    """
    assertError(error)
    return collectDetectionsResult(detectionsBatch, images)


class HumanDetector:
    """
    Human detector. Human is optional Union face, body, ...

    Attributes:
        _detector (IDetectorPtr): core detector
        _launchOptions (LaunchOptions): detector launch options
    """

    __slots__ = ("_detector", "_launchOptions")

    def __init__(self, detectorPtr, launchOptions: LaunchOptions):
        self._detector = detectorPtr
        self._launchOptions = launchOptions

    @property
    def launchOptions(self) -> LaunchOptions:
        """Get detector launch options"""
        return self._launchOptions

    @overload  # type: ignore
    def detect(
        self,
        images: List[Union[VLImage, ImageForDetection]],
        asyncEstimate: Literal[False] = False,
    ) -> HumanDetectResult:
        ...

    @overload
    def detect(
        self,
        images: List[Union[VLImage, ImageForDetection]],
        asyncEstimate: Literal[True],
    ) -> AsyncTask[HumanDetectResult]:
        ...

    def detect(
        self,
        images: List[Union[VLImage, ImageForDetection]],
        asyncEstimate=False,
    ) -> Union[HumanDetectResult, AsyncTask[HumanDetectResult]]:
        """
        Batch detect humans on images.

        Args:
            images: input images list. Format must be R8G8B8
            asyncEstimate: estimate or run estimation in background
        Returns:
            asyncEstimate is False: return list of lists detection, order of detection lists
                                    is corresponding to order input images
            asyncEstimate is True: async task
        Raises:
            LunaSDKException if an error occurs
        """
        coreImages, detectAreas = getArgsForCoreDetectorForImages(images)
        validateBatchDetectInput(self._detector, coreImages, detectAreas)
        if asyncEstimate:
            task = self._detector.asyncDetect(coreImages, detectAreas)
            return AsyncTask(task, postProcessing=partial(postProcessing, images=images))
        error, fsdkDetectRes = self._detector.detect(coreImages, detectAreas)
        return postProcessing(error, fsdkDetectRes, images=images)
