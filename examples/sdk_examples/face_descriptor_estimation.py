"""
Face descriptor estimate example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.facedetector import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def estimateDescriptor():
    """
    Estimate face descriptor.
    """
    image = VLImage.load(filename='C:/temp/test.jpg')
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    faceDetection = detector.detectOne(image)
    warper = faceEngine.createWarper()
    warp = warper.warp(faceDetection)

    extractor = faceEngine.createFaceDescriptorEstimator()

    pprint.pprint(extractor.estimate(warp.warpedImage))
    pprint.pprint(extractor.estimateDescriptorsBatch([warp.warpedImage, warp.warpedImage]))
    batch, aggregateDescriptor = extractor.estimateDescriptorsBatch([warp.warpedImage, warp.warpedImage],
                                                                    aggregate=True)
    pprint.pprint(batch)
    pprint.pprint(aggregateDescriptor)


if __name__ == "__main__":
    estimateDescriptor()
