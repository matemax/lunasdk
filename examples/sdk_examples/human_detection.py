"""
Module realize simple examples following features:
    * one human detection
    * batch images human detection
    * detect landmarks17
"""
import asyncio
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.detectors.base import ImageForDetection
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_WITHOUT_FACES, EXAMPLE_SEVERAL_FACES, EXAMPLE_O


def detectHumanBody():
    """
    Detect one human body on an image.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()

    imageWithOneHuman = VLImage.load(filename=EXAMPLE_O)
    pprint.pprint(detector.detectOne(imageWithOneHuman, detectLandmarks=False).asDict())
    imageWithSeveralHumans = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    pprint.pprint(detector.detectOne(imageWithSeveralHumans, detectLandmarks=False).asDict())

    severalHumans = detector.detect([imageWithSeveralHumans], detectLandmarks=True)
    pprint.pprint([human.asDict() for human in severalHumans[0]])

    imageWithoutHuman = VLImage.load(filename=EXAMPLE_WITHOUT_FACES)
    pprint.pprint(detector.detectOne(imageWithoutHuman, detectLandmarks=False) is None)

    severalHumans = detector.detect([ImageForDetection(imageWithSeveralHumans, Rect(1, 1, 300.0, 300.0))])
    pprint.pprint(severalHumans)


async def asyncDetectHumanBody():
    """
    Async detect one human body on an image.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()

    imageWithOneHuman = VLImage.load(filename=EXAMPLE_O)
    pprint.pprint(detector.detectOne(imageWithOneHuman, detectLandmarks=False).asDict())
    imageWithSeveralHumans = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    human = await detector.detectOne(imageWithSeveralHumans, detectLandmarks=False, asyncEstimate=True)
    pprint.pprint(human.asDict())

    task1 = detector.detect([imageWithSeveralHumans], detectLandmarks=True, asyncEstimate=True)
    task2 = detector.detect([imageWithSeveralHumans], detectLandmarks=True, asyncEstimate=True)
    for task in (task1, task2):
        pprint.pprint(task.get())


if __name__ == "__main__":
    detectHumanBody()
    asyncio.run(asyncDetectHumanBody())
