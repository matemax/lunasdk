"""
An emotion estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O


def estimateBasicAttributes():
    """
    Estimate emotion from a warped image.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createWarper()
    warp = warper.warp(faceDetection)

    basicAttributesEstimator = faceEngine.createBasicAttributesEstimator()

    pprint.pprint(
        basicAttributesEstimator.estimate(
            warp.warpedImage, estimateAge=True, estimateGender=True, estimateEthnicity=True
        ).asDict()
    )


if __name__ == "__main__":
    estimateBasicAttributes()
