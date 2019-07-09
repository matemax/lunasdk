import unittest

from lunavl.sdk.faceengine.engine import VLFaceEngine


class BaseTestClass(unittest.TestCase):
    faceEngine: VLFaceEngine = None

    @classmethod
    def setup_class(cls):
        cls.faceEngine = VLFaceEngine()

