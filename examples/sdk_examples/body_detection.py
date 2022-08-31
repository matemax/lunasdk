"""
Module realize simple examples following features:
    * one human detection
    * batch images human detection
    * detect landmarks17
"""
import asyncio
import pprint

from resources import EXAMPLE_O, EXAMPLE_SEVERAL_FACES, EXAMPLE_WITHOUT_FACES

from lunavl.sdk.detectors.base import ImageForDetection
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage


def detectHumanBody():
    """
    Detect one human body on an image.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createBodyDetector()

    imageWithOneBody = VLImage.load(filename=EXAMPLE_O)
    pprint.pprint(detector.detectOne(imageWithOneBody, detectLandmarks=False).asDict())
    imageWithSeveralBodies = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    pprint.pprint(detector.detectOne(imageWithSeveralBodies, detectLandmarks=False).asDict())

    severalBodies = detector.detect([imageWithSeveralBodies], detectLandmarks=True)
    pprint.pprint([human.asDict() for human in severalBodies[0]])

    imageWithoutBody = VLImage.load(filename=EXAMPLE_WITHOUT_FACES)
    pprint.pprint(detector.detectOne(imageWithoutBody, detectLandmarks=False) is None)

    severalBodies = detector.detect([ImageForDetection(imageWithSeveralBodies, Rect(1, 1, 300.0, 300.0))])
    pprint.pprint(severalBodies)


async def asyncDetectHumanBody():
    """
    Async detect one human body on an image.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createBodyDetector()

    imageWithOneBody = VLImage.load(filename=EXAMPLE_O)
    pprint.pprint(detector.detectOne(imageWithOneBody, detectLandmarks=False).asDict())
    imageWithSeveralBodies = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    human = await detector.detectOne(imageWithSeveralBodies, detectLandmarks=False, asyncEstimate=True)
    pprint.pprint(human.asDict())

    task1 = detector.detect([imageWithSeveralBodies], detectLandmarks=True, asyncEstimate=True)
    task2 = detector.detect([imageWithSeveralBodies], detectLandmarks=True, asyncEstimate=True)
    for task in (task1, task2):
        pprint.pprint(task.get())


if __name__ == "__main__":
    detectHumanBody()
    asyncio.run(asyncDetectHumanBody())
