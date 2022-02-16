"""
An emotion estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_SEVERAL_FACES


def estimateBasicAttributes():
    """
    Estimate emotion from a warped image.
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


if __name__ == "__main__":
    estimateBasicAttributes()
