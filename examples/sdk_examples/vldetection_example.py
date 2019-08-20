"""
Example of using VLFaceDetector
"""
import pprint
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.luna_faces import VLFaceDetector


def estimateAll():
    """
    Estimate all attributes
    """
    detector = VLFaceDetector()
    image = VLImage.load(filename="C:/temp/test.jpg")
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


if __name__ == "__main__":
    estimateAll()
