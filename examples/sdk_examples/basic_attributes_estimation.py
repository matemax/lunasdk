"""
An emotion estimation example
"""
import asyncio
import pprint

from resources import EXAMPLE_SEVERAL_FACES

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateBasicAttributes():
    """
    Estimate basic attributes.
    """
    image = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetections = detector.detect([image])[0]
    warper = faceEngine.createFaceWarper()
    warps = [warper.warp(faceDetection) for faceDetection in faceDetections]

    basicAttributesEstimator = faceEngine.createBasicAttributesEstimator()

    pprint.pprint(
        basicAttributesEstimator.estimate(
            warps[0].warpedImage, estimateAge=True, estimateGender=True, estimateEthnicity=True
        ).asDict()
    )

    pprint.pprint(
        basicAttributesEstimator.estimateBasicAttributesBatch(
            warps, estimateAge=True, estimateGender=True, estimateEthnicity=True
        )
    )

    pprint.pprint(
        basicAttributesEstimator.estimateBasicAttributesBatch(
            warps, estimateAge=True, estimateGender=True, estimateEthnicity=True, aggregate=True
        )
    )


async def asyncEstimateBasicAttributes():
    """
    Async estimate basic attributes.
    """
    image = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    faceDetections = detector.detect([image])[0]
    warper = faceEngine.createFaceWarper()
    warps = [warper.warp(faceDetection) for faceDetection in faceDetections]

    basicAttributesEstimator = faceEngine.createBasicAttributesEstimator()
    basicAttributes = await basicAttributesEstimator.estimate(
        warps[0].warpedImage, estimateAge=True, estimateGender=True, estimateEthnicity=True, asyncEstimate=True
    )
    pprint.pprint(basicAttributes.asDict())

    task1 = basicAttributesEstimator.estimate(
        warps[0].warpedImage, estimateAge=True, estimateGender=True, estimateEthnicity=True, asyncEstimate=True
    )
    task2 = basicAttributesEstimator.estimate(
        warps[0].warpedImage, estimateAge=True, estimateGender=True, estimateEthnicity=True, asyncEstimate=True
    )
    for task in (task1, task2):
        pprint.pprint(task.get().asDict())


if __name__ == "__main__":
    estimateBasicAttributes()
    asyncio.run(asyncEstimateBasicAttributes())
