"""
Module realize simple examples following features:
    * one face detection
    * batch images face detection
    * detect landmarks68 and landmarks5
"""
import asyncio
from time import perf_counter

from resources import EXAMPLE_O

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.launch_options import DeviceClass, LaunchOptions

COUNT = 1000
BATCH_SIZE = 10
image = EXAMPLE_O

faceEngine = VLFaceEngine()


# load image to cpu and run estimator on gpu
async def asyncDetectFacesImageCpu():
    """
    Async detect faces on images.
    """
    images = [VLImage.load(filename=image) for _ in range(BATCH_SIZE)]

    detector = faceEngine.createFaceDetector(
        DetectorType.FACE_DET_V3, launchOptions=LaunchOptions(deviceClass=DeviceClass.gpu)
    )

    await detector.detect(images, asyncEstimate=True)
    start = perf_counter()
    for _ in range(COUNT):
        await detector.detect(images, asyncEstimate=True)
    print("cpu", perf_counter() - start)


# load image to gpu and run estimator on gpu
async def asyncDetectFacesImageGpu():
    """
    Async detect faces on images.
    """
    images = [VLImage.load(filename=image) for _ in range(BATCH_SIZE)]
    detector = faceEngine.createFaceDetector(
        DetectorType.FACE_DET_V3, launchOptions=LaunchOptions(deviceClass=DeviceClass.gpu)
    )
    await detector.detect(images, asyncEstimate=True)
    start = perf_counter()
    for _ in range(COUNT):
        await detector.detect(images, asyncEstimate=True)
    print("gpu", perf_counter() - start)


if __name__ == "__main__":
    asyncio.run(asyncDetectFacesImageCpu())
    asyncio.run(asyncDetectFacesImageGpu())
    asyncio.run(asyncDetectFacesImageCpu())
    asyncio.run(asyncDetectFacesImageGpu())
