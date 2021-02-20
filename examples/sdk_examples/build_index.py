"""Module realize simple examples following features:
    * build index with descriptors
    * search for descriptors with the shorter distance to passed descriptor
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O, EXAMPLE_1


def buildDescriptorIndex():
    """
    Build index and search.
    """
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
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
    result = index.search(faceDescriptor, 1)
    pprint.pprint(f"result: {result}")


if __name__ == "__main__":
    buildDescriptorIndex()
