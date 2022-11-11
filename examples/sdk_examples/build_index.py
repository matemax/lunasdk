"""Module realize simple examples following features:
    * build index with descriptors
    * search for descriptors with the shorter distance to passed descriptor
"""
import asyncio
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType, FaceEngineSettingsProvider
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_1, EXAMPLE_O


def buildDescriptorIndex():
    """
    Build index and search.
    """

    feConf = FaceEngineSettingsProvider()
    feConf.index.construction = 1200
    faceEngine = VLFaceEngine(faceEngineConf=feConf)

    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    warper = faceEngine.createFaceWarper()
    extractor = faceEngine.createFaceDescriptorEstimator()
    descriptorsBatch = faceEngine.createFaceDescriptorFactory().generateDescriptorsBatch(2)

    for image in (EXAMPLE_O, EXAMPLE_1):
        vlImage = VLImage.load(filename=image)
        faceDetection = detector.detectOne(vlImage)
        warp = warper.warp(faceDetection)
        faceDescriptor = extractor.estimate(warp.warpedImage)
        descriptorsBatch.append(faceDescriptor)

    indexBuilder = faceEngine.createIndexBuilder()
    indexBuilder.appendBatch(descriptorsBatch)
    pprint.pprint(f"index buf size: {indexBuilder.bufSize}")
    index = indexBuilder.buildIndex()
    pprint.pprint(index[0])
    index.remove(0)
    result = index.search(faceDescriptor, 1)
    pprint.pprint(f"result: {result}")

    index.save("dynamic-index.dat")


async def asyncBuildDescriptorIndex():
    """
    Async search by index.
    """
    """
    Build index and search.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    warper = faceEngine.createFaceWarper()
    extractor = faceEngine.createFaceDescriptorEstimator()
    descriptorsBatch = faceEngine.createFaceDescriptorFactory().generateDescriptorsBatch(2)

    for image in (EXAMPLE_O, EXAMPLE_1):
        vlImage = VLImage.load(filename=image)
        faceDetection = await detector.detectOne(vlImage, asyncEstimate=True)
        warp = warper.warp(faceDetection)
        faceDescriptor = await extractor.estimate(warp.warpedImage, asyncEstimate=True)
        descriptorsBatch.append(faceDescriptor)

    indexBuilder = faceEngine.createIndexBuilder()
    indexBuilder.appendBatch(descriptorsBatch)
    index = indexBuilder.buildIndex()
    pprint.pprint(index[0])
    result = await index.search(faceDescriptor, 2, asyncSearch=True)
    pprint.pprint(f"result: {result}")


if __name__ == "__main__":
    buildDescriptorIndex()
    asyncio.run(asyncBuildDescriptorIndex())
