"""
Face descriptor estimate example
"""

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage


def matchDescriptors():
    """
    Estimate face descriptor.
    """

    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    extractor = faceEngine.createFaceDescriptorEstimator()
    warper = faceEngine.createWarper()
    matcher = faceEngine.createFaceMatcher()

    image1 = VLImage.load(filename="C:/temp/test.jpg")

    faceDetection1 = detector.detectOne(image1)
    warp1 = warper.warp(faceDetection1)
    descriptor1 = extractor.estimate(warp1.warpedImage)

    image2 = VLImage.load(filename="C:/temp/female_caucasian_warp.jpg")
    faceDetection2 = detector.detectOne(image2)
    warp2 = warper.warp(faceDetection2)
    descriptor2 = extractor.estimate(warp2.warpedImage)
    batch, _ = extractor.estimateDescriptorsBatch([warp1.warpedImage, warp2.warpedImage])

    print(matcher.match(descriptor1, descriptor2))
    print(matcher.match(descriptor1, batch))
    print(matcher.match(descriptor1, [descriptor2, descriptor1]))


if __name__ == "__main__":
    matchDescriptors()
