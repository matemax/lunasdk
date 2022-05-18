"""
Example of using VLFaceDetector
"""
import pprint

from lunavl.sdk.estimator_collections import EstimatorsSettings, FaceDescriptorEstimatorSettings
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.launch_options import LaunchOptions, DeviceClass
from lunavl.sdk.luna_faces import VLFaceDetector, FaceDetectorSettings
from resources import EXAMPLE_O


def estimateAll():
    """
    Estimate all attributes
    """
    estimatorsSettings = EstimatorsSettings(
        descriptor=FaceDescriptorEstimatorSettings(
            descriptorVersion=59, launchOptions=LaunchOptions(deviceClass=DeviceClass.cpu)
        )
    )
    detectorSettings = FaceDetectorSettings(launchOptions=LaunchOptions(deviceClass=DeviceClass.cpu))
    VLFaceDetector.initialize(estimatorsSettings=estimatorsSettings)

    detector = VLFaceDetector(detectorSettings=detectorSettings)
    image = VLImage.load(filename=EXAMPLE_O)
    detection = detector.detectOne(image)
    pprint.pprint(detection.basicAttributes.asDict())
    pprint.pprint(detection.emotions.asDict())
    pprint.pprint(detection.warpQuality.asDict())
    pprint.pprint(detection.eyes.asDict())
    pprint.pprint(detection.gaze.asDict())
    pprint.pprint(detection.headPose.asDict())
    pprint.pprint(detection.mouthState.asDict())
    pprint.pprint(detection.ags)
    pprint.pprint(detection.descriptor.asDict())
    pprint.pprint(detection.liveness.asDict())


if __name__ == "__main__":
    estimateAll()
