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

    def createDetector(self, detectorType: DetectorType) -> FaceDetector:
        return FaceDetector(self._faceEngine.createDetector(detectorType.coreDetectorType), detectorType)

FACE_ENGINE = VLFaceEngine()
