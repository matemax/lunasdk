"""
Module realize simple examples following features:
    * redetect one face detection
    * redetect several faces
"""
import asyncio
import pprint

from lunavl.sdk.detectors.base import ImageForRedetection
from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_SEVERAL_FACES, EXAMPLE_O


def detectFaces():
    """
    Redetect faces on images.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)

    imageWithOneFace = VLImage.load(filename=EXAMPLE_O)
    pprint.pprint(detector.detectOne(imageWithOneFace, detect5Landmarks=False, detect68Landmarks=False).asDict())
    detection = detector.detectOne(imageWithOneFace, detect5Landmarks=False, detect68Landmarks=False)
    pprint.pprint(detector.redetectOne(image=imageWithOneFace, bBox=detection))
    pprint.pprint(detector.redetectOne(image=imageWithOneFace, bBox=detection.boundingBox.rect))

    imageWithSeveralFaces = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    severalFaces = detector.detect([imageWithSeveralFaces], detect5Landmarks=False, detect68Landmarks=False)

    pprint.pprint(
        detector.redetect(
            images=[
                ImageForRedetection(imageWithSeveralFaces, [face.boundingBox.rect for face in severalFaces[0]]),
                ImageForRedetection(imageWithOneFace, [detection.boundingBox.rect]),
                ImageForRedetection(imageWithOneFace, [Rect(0, 0, 1, 1)]),
            ]
        )
    )


async def asyncRedetectFaces():
    """
    Async redetect faces on images.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)

    imageWithSeveralFaces = VLImage.load(filename=EXAMPLE_SEVERAL_FACES)
    severalFaces = detector.detect([imageWithSeveralFaces], detect5Landmarks=False, detect68Landmarks=False)

    detections = await detector.redetect(
        images=[
            ImageForRedetection(imageWithSeveralFaces, [face.boundingBox.rect for face in severalFaces[0]]),
        ],
        asyncEstimate=True,
    )
    pprint.pprint(detections)
    task1 = detector.redetect(
        images=[
            ImageForRedetection(imageWithSeveralFaces, [severalFaces[0][0].boundingBox.rect]),
        ],
        asyncEstimate=True,
    )
    task2 = detector.redetect(
        images=[
            ImageForRedetection(imageWithSeveralFaces, [severalFaces[0][1].boundingBox.rect]),
        ],
        asyncEstimate=True,
    )
    for task in (task1, task2):
        pprint.pprint(task.get())


if __name__ == "__main__":
    detectFaces()
    asyncio.run(asyncRedetectFaces())
