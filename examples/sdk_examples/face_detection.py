"""
Module realize simple examples following features:
    * one face detection
    * batch images face detection
    * detect landmarks68 and landmarks5
"""

import asyncio
import os
import pprint
import sys
from time import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from lunavl.sdk.faceengine.engine import VLFaceEngine  # noqa: E402
from lunavl.sdk.faceengine.setting_provider import DetectorType  # noqa: E402
from lunavl.sdk.image_utils.image import VLImage  # noqa: E402
from resources import EXAMPLE_O  # noqa: E402


async def detectFaces():
    """
    Detect one face on an image.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)

    imageWithOneFace = VLImage.load(filename=EXAMPLE_O)
    pprint.pprint((detector.detectOne(imageWithOneFace, detect5Landmarks=False, detect68Landmarks=False)).asDict())

    count = 1000
    conc = 2

    async def worker(iterator):
        for _ in iterator:
            await detector.aDetectOne(imageWithOneFace, detect5Landmarks=False, detect68Landmarks=False)

    it = iter((i for i in range(count)))
    start = time()
    await asyncio.gather(*[worker(it) for _ in range(conc)])
    print(time() - start)

    start = time()
    for _ in range(count):
        detector.detectOne(imageWithOneFace, detect5Landmarks=False, detect68Landmarks=False)
    print(time() - start)


if __name__ == "__main__":
    asyncio.run(detectFaces())
