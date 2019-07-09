from collections import namedtuple
from pathlib import Path

import numpy as np
import FaceEngine as fe

from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ONE_FACE
from PIL import Image


class TestImage(BaseTestClass):
    """
    Test of detector.
    """

    def test_image_initialize(self):
        path = Path(ONE_FACE)
        with path.open("rb") as file:
            binaryBody = file.read()
        img = Image.open(ONE_FACE)
        npBody = np.asarray(img)
        coreBody = fe.Image()
        coreBody.load(ONE_FACE)
        InitCase = namedtuple("InitCase", ("initType", "body"))
        cases = (InitCase("bytes", binaryBody), InitCase("numpy array", npBody), InitCase("core", coreBody))
        for case in cases:
            with self.subTest(initType=case.initType):
                image = VLImage(body=case.body, filename=case.initType)
                assert case.initType == image.filename
                assert image.isValid()
                assert image.rect == Rect(0, 0, 912, 1080)

    def test_image_load_from_file(self):
        image = VLImage.load(filename=ONE_FACE)
        assert image.rect == Rect(0, 0, 912, 1080)
        assert image.isValid()
        assert image.source == 'one_face.jpg'

    def test_image_load_from_url(self):
        url = 'https://cdn1.savepice.ru/uploads/2019/4/15/194734af15c4fcd06dec6db86bbeb7cd-full.jpg'
        image = VLImage.load(url=url)
        assert image.isValid()
        assert image.rect == Rect(0, 0, 497, 640)
        assert image.source == url
