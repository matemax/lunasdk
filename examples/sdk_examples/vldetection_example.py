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
    image = VLImage.load(
        url='https://cdn1.savepice.ru/uploads/2019/4/15/aa970957128d9892f297cdfa5b3fda88-full.jpg')
    detection = detector.detectOne(image)
    pprint.pprint(detection.basicAttributes.asDict())
    pprint.pprint(detection.emotions.asDict())
    pprint.pprint(detection.warpQuality.asDict())
    pprint.pprint(detection.eyes.asDict())
    pprint.pprint(detection.gaze.asDict())
    pprint.pprint(detection.headPose.asDict())
    pprint.pprint(detection.mouthState.asDict())
    pprint.pprint(detection.ags)


if __name__ == "__main__":
    estimateAll()
