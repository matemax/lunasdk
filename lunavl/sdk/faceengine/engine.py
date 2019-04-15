"""
Module realize wraps on facengine objects
"""
import os
from typing import Optional

import FaceEngine as CoreFE

from lunavl.sdk.faceengine.facedetector import DetectorType, FaceDetector, ImageForDetection
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage


class VLFaceEngine:
    """
    Wraper on FaceEngine.

    Attributes:
        dataPath (str): path to a faceengine data folder
        configPath (str): path to a faceengine configuration file
        _faceEngine (PyIFaceEngine): python C++ binding on IFaceEngine, Root LUNA SDK object interface
    """

    def __init__(self, pathToData: Optional[str] = None, pathToFaceEngineConf: Optional[str] = None):
        """
        Init.

        Args:
            pathToData: path to a faceengine data folder
            pathToFaceEngineConf:  path to a faceengine configuration file
        """
        if pathToData is None:
            if "FSDK_ROOT" in os.environ:
                pathToData = os.path.join(os.environ["FSDK_ROOT"], "data")
            else:
                raise ValueError("Failed on path to faceengine luna data folder, set variable pathToData or set"
                                 "environment variable *FSDK_ROOT*")
            if pathToFaceEngineConf is None:
                if "FSDK_ROOT" in os.environ:
                    pathToFaceEngineConf = os.path.join(os.environ["FSDK_ROOT"], "data", "faceengine.conf")
                # else:
                #     raise ValueError("Failed on path to faceengine luna data folder, set variable pathToFaceEngineConf"
                #                      " or set environment variable *FSDK_ROOT*")
        self.dataPath = pathToData
        self.configPath = CoreFE.createSettingsProvider(pathToFaceEngineConf)
        # todo: validate initialize
        self._faceEngine = CoreFE.createFaceEngine(dataPath=pathToData, configPath=pathToFaceEngineConf)

    def createFaceDetector(self, detectorType: DetectorType) -> FaceDetector:
        """
        Create face detector.

        Args:
            detectorType: detector type

        Returns:
            detector
        """
        return FaceDetector(self._faceEngine.createFaceDetector(detectorType.coreDetectorType), detectorType)

# (VLFaceEngine): Global
FACE_ENGINE = VLFaceEngine()


if __name__ == "__main__":
    fe = VLFaceEngine()
    d = fe.createFaceDetector(DetectorType.FACE_DET_V1)
    image1 = VLImage.load(filename="C:/temp/test.jpg")
    image2 = VLImage.load(filename="C:/temp/kand.jpg")
    image3 = VLImage.load(filename="C:/temp/multiple_faces.jpg")
    detections = d.detect([image1, image2, ImageForDetection(image3, Rect(width=1600, height= 1000))])
    import pprint
    pprint.pprint([[detection.asDict() for detection in imageDetections] for imageDetections in detections])
