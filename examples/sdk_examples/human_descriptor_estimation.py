"""
Human descriptor estimate example
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O


def estimateDescriptor():
    """
    Estimate human descriptor.
    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createHumanDetector()
    humanDetection = detector.detectOne(image)
    warper = faceEngine.createHumanWarper()
    warp = warper.warp(humanDetection)

    extractor = faceEngine.createHumanDescriptorEstimator()

    pprint.pprint(extractor.estimate(warp.warpedImage))
    pprint.pprint(extractor.estimateDescriptorsBatch([warp.warpedImage, warp.warpedImage]))
    batch, aggregateDescriptor = extractor.estimateDescriptorsBatch(
        [warp.warpedImage, warp.warpedImage], aggregate=True
    )
    pprint.pprint(batch)
    pprint.pprint(aggregateDescriptor)


if __name__ == "__main__":
    estimateDescriptor()
