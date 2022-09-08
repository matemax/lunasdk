"""
Module realize simple examples following features:
    * detect humans
    * async detect humans
"""
import asyncio
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O, EXAMPLE_SEVERAL_FACES


def detectHumans():
    """
    Detect humans on an image.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()

    imageWithSeveralFaces = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)

    severalHumans = detector.detect([imageWithSeveralFaces])
    pprint.pprint([human.asDict() for human in severalHumans[0]])


async def asyncHumans():
    """
    Async detect faces on images.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()

    image1 = VLImage.load(filename=EXAMPLE_O)
    image2 = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    detections = await detector.detect([image1], asyncEstimate=True)
    pprint.pprint([human.asDict() for human in detections[0]])
    detections = await detector.detect(images=[image1], asyncEstimate=True)
    pprint.pprint([human.asDict() for human in detections[0]])
    task1 = detector.detect(
        images=[image1],
        asyncEstimate=True,
    )
    task2 = detector.detect(
        images=[
            image2,
        ],
        asyncEstimate=True,
    )
    for task in (task1, task2):
        pprint.pprint([human.asDict() for human in task.get()[0]])


if __name__ == "__main__":
    detectHumans()
    asyncio.run(asyncHumans())
