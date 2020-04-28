"""
Module realize simple examples following features:
    * redect one human detection
    * redect several humans
"""
import pprint

from lunavl.sdk.detectors.base import ImageForRedetection
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_SEVERAL_FACES, EXAMPLE_O


def detectHumans():
    """
    Redect human body on images.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()

    imageWithOneHuman = VLImage.load(filename=EXAMPLE_O)
    detection = detector.detectOne(imageWithOneHuman, detectLandmarks=False)
    pprint.pprint(detector.redetectOne(image=imageWithOneHuman, detection=detection))
    pprint.pprint(detector.redetectOne(image=imageWithOneHuman, bBox=detection.boundingBox.rect))

    imageWithSeveralHumans = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    severalHumans = detector.detect([imageWithSeveralHumans], detectLandmarks=False)

    pprint.pprint(
        detector.redetect(
            images=[
                ImageForRedetection(imageWithSeveralHumans, [human.boundingBox.rect for human in severalHumans[0]]),
                ImageForRedetection(imageWithOneHuman, [detection.boundingBox.rect]),
                ImageForRedetection(imageWithOneHuman, [Rect(0, 0, 1, 1)]),
            ]
        )
    )


if __name__ == "__main__":
    detectHumans()
