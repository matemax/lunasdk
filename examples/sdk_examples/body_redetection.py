"""
Module realize simple examples following features:
    * redetect one human detection
    * redetect several humans
"""
import asyncio
import pprint

from resources import EXAMPLE_O, EXAMPLE_SEVERAL_FACES

from lunavl.sdk.detectors.base import ImageForRedetection
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage


def detectBodies():
    """
    Redetect human body on images.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createBodyDetector()

    imageWithOneBody = VLImage.load(filename=EXAMPLE_O)
    detection = detector.detectOne(imageWithOneBody, detectLandmarks=False)
    pprint.pprint(detector.redetectOne(image=imageWithOneBody, bBox=detection))
    pprint.pprint(detector.redetectOne(image=imageWithOneBody, bBox=detection.boundingBox.rect))

    imageWithSeveralBodies = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    severalBodies = detector.detect([imageWithSeveralBodies], detectLandmarks=False)

    pprint.pprint(
        detector.redetect(
            images=[
                ImageForRedetection(imageWithSeveralBodies, [human.boundingBox.rect for human in severalBodies[0]]),
                ImageForRedetection(imageWithOneBody, [detection.boundingBox.rect]),
                ImageForRedetection(imageWithOneBody, [Rect(0, 0, 100, 100)]),
            ]
        )
    )


async def asyncRedetectBodies():
    """
    Async redetect human body on images.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createBodyDetector()

    imageWithOneBody = VLImage.load(filename=EXAMPLE_O)
    detection = detector.detectOne(imageWithOneBody, detectLandmarks=False)
    detection = await detector.redetectOne(image=imageWithOneBody, bBox=detection, asyncEstimate=True)
    imageWithSeveralBodies = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    severalBodies = detector.detect([imageWithSeveralBodies], detectLandmarks=False)

    task1 = detector.redetect(
        images=[
            ImageForRedetection(imageWithSeveralBodies, [human.boundingBox.rect for human in severalBodies[0]]),
        ],
        asyncEstimate=True,
    )
    task2 = detector.redetect(
        images=[
            ImageForRedetection(imageWithOneBody, [detection.boundingBox.rect]),
        ],
        asyncEstimate=True,
    )
    for task in (task1, task2):
        pprint.pprint(task.get())


if __name__ == "__main__":
    detectBodies()
    asyncio.run(asyncRedetectBodies())
