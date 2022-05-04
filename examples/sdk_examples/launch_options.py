"""
Module realize simple examples following features:
    * one face detection
    * batch images face detection
    * detect landmarks68 and landmarks5
"""
import asyncio
from time import perf_counter, sleep

from lunavl.sdk.faceengine.engine import VLFaceEngine, LaunchOptions
from lunavl.sdk.faceengine.setting_provider import DetectorType, DeviceClass
from lunavl.sdk.image_utils.image import VLImage, TargetDevice
from resources import EXAMPLE_SEVERAL_FACES, EXAMPLE_O, EXAMPLE_1, EXAMPLE_2

COUNT = 1000
BATCH_SIZE= 10
image = EXAMPLE_O

faceEngine = VLFaceEngine()
# load image to cpu and run estimator on gpu
async def asyncDetectFacesImageCpu():
    """
    Async detect faces on images.
    """
    images = [VLImage.load(filename=image, device=TargetDevice.cpu) for _ in range(BATCH_SIZE)]

    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3,
                                             launchOptions=LaunchOptions(deviceClass=DeviceClass.gpu))


    detections = await detector.detect(images, asyncEstimate=True)
    start = perf_counter()
    for _ in range(COUNT):
        detections = await detector.detect(images, asyncEstimate=True)
    print("cpu", perf_counter() - start)

# load image to gpu and run estimator on gpu
async def asyncDetectFacesImageGpu():
    """
    Async detect faces on images.
    """
    images = [VLImage.load(filename=image, device=TargetDevice.gpu) for _ in range(BATCH_SIZE)]
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3,
                                             launchOptions=LaunchOptions(deviceClass=DeviceClass.gpu))
    detections = await detector.detect(images, asyncEstimate=True)
    start = perf_counter()
    for _ in range(COUNT):
        detections = await detector.detect(images, asyncEstimate=True)
    print("gpu", perf_counter() - start)


if __name__ == "__main__":
    asyncio.run(asyncDetectFacesImageCpu())
    asyncio.run(asyncDetectFacesImageGpu())
    asyncio.run(asyncDetectFacesImageCpu())
    asyncio.run(asyncDetectFacesImageGpu())

