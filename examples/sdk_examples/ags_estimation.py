"""
An approximate garbage score estimation example
"""
import pprint

from resources import EXAMPLE_O, EXAMPLE_1
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateAGS():
    """
    Estimate face detection ags.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)

    agsEstimator = faceEngine.createAGSEstimator()

    pprint.pprint(agsEstimator.estimate(image=image, boundingBox=faceDetection.boundingBox))
    pprint.pprint(agsEstimator.estimate(faceDetection))

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetections = detector.detect([image, image2])
    boundingBoxes = [detection[0].boundingBox for detection in faceDetections]

    pprint.pprint(agsEstimator.estimateAgsBatch(images=[image, image2], boundingBoxes=boundingBoxes))


if __name__ == "__main__":
    estimateAGS()
