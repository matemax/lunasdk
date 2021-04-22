"""
Module realize simple examples following features:
    * head pose estimation
    * batch images face detection
    * detect landmarks68 and landmarks5
"""
import pprint

from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from resources import EXAMPLE_O, EXAMPLE_1


def estimateHeadPose():
    """
    Example of a head pose estimation.

    """
    image = VLImage.load(filename=EXAMPLE_O)
    faceEngine = VLFaceEngine()
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V1)
    headPoseEstimator = faceEngine.createHeadPoseEstimator()
    faceDetection = detector.detectOne(image, detect5Landmarks=False, detect68Landmarks=True)
    #: estimate by 68 landmarks
    angles = headPoseEstimator.estimateBy68Landmarks(faceDetection.landmarks68)
    pprint.pprint(angles.asDict())

    #: get frontal type
    pprint.pprint(angles.getFrontalType())

    #: estimate by detection
    angles = headPoseEstimator.estimateByBoundingBox(faceDetection.boundingBox, image)
    angles.getFrontalType()
    pprint.pprint(angles)

    image2 = VLImage.load(filename=EXAMPLE_1)
    faceDetection2 = detector.detectOne(image2, detect5Landmarks=False, detect68Landmarks=True)
    boundingBoxes = [detection.boundingBox for detection in (faceDetection, faceDetection2)]
    #: batch estimate by detection
    anglesList = headPoseEstimator.estimateByBoundingBoxBatch(boundingBoxes, [image, image2])
    pprint.pprint(anglesList)


if __name__ == "__main__":
    estimateHeadPose()
