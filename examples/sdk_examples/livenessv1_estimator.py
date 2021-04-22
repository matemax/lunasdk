"""
LivenessV1 estimation example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O, EXAMPLE_1


def estimateLiveness():
    """
    Estimate liveness.
    """

    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image, detect68Landmarks=True)

    livenessEstimator = faceEngine.createLivenessV1Estimator()

    pprint.pprint(livenessEstimator.estimate(faceDetection, qualityThreshold=0.5).asDict())

    faceDetection2 = detector.detectOne(VLImage.load(filename=EXAMPLE_1), detect68Landmarks=True)
    pprint.pprint(livenessEstimator.estimateBatch([faceDetection, faceDetection2]))


if __name__ == "__main__":
    estimateLiveness()
