"""
Example of using VLFaceDetector
"""
import pprint
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.luna_faces import VLFaceDetector
from lunavl.sdk.faceengine.engine import VLFaceEngine
from resources import EXAMPLE_O


def estimateAll():
    """
    Estimate all attributes
    """
    VLFaceDetector.initialize(VLFaceEngine())
    detector = VLFaceDetector()
    image = VLImage.load(filename=EXAMPLE_O)
    detection = detector.detectOne(image)
    pprint.pprint(detection.basicAttributes.asDict())
    pprint.pprint(detection.emotions.asDict())
    pprint.pprint(detection.warpQuality.asDict())
    pprint.pprint(detection.eyes.asDict())
    pprint.pprint(detection.gaze.asDict())
    pprint.pprint(detection.headPose.asDict())
    pprint.pprint(detection.mouthState.asDict())
    pprint.pprint(detection.ags)
    pprint.pprint(detection.descriptor.asDict())
    pprint.pprint(detection.liveness.asDict())
    pprint.pprint(detection.orientationMode.name)


if __name__ == "__main__":
    estimateAll()
