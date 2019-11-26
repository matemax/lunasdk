from collections import namedtuple
from pathlib import Path
import pytest
import numpy as np
import FaceEngine as fe

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage, ColorFormat
from tests.base import BaseTestClass
from tests.resources import ONE_FACE
from PIL import Image


class TestImage(BaseTestClass):
    """
    Test of image.
    """

    def test_image_initialize(self):
        path = Path(ONE_FACE)
        with path.open("rb") as file:
            binaryBody = file.read()
        byteBody = bytearray(binaryBody)
        img = Image.open(ONE_FACE)
        npBody = np.asarray(img)
        coreBody = fe.Image()
        coreBody.load(ONE_FACE)
        InitCase = namedtuple("InitCase", ("initType", "body"))
        cases = (InitCase("bytes", binaryBody), InitCase("numpy array", npBody), InitCase("byte array", byteBody),
                 InitCase("core", coreBody))
        for case in cases:
            with self.subTest(initType=case.initType):
                image = VLImage(body=case.body, filename=case.initType)
                assert image.isValid()
                assert case.initType == image.filename
                assert image.rect == Rect(0, 0, 912, 1080)

    def test_image_load_from_file(self):
        image = VLImage.load(filename=ONE_FACE)
        assert image.isValid()
        assert image.rect == Rect(0, 0, 912, 1080)
        assert image.filename == "one_face.jpg"

    def test_image_load_from_url(self):
        url = "https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg"
        image = VLImage.load(url=url)
        assert image.isValid()
        assert image.rect == Rect(0, 0, 1000, 1288)
        assert image.filename == url

    def test_image_bad_filename_or_url(self):
        url = "https://st.kp.yandex.net/images/"
        filename = 'test.jpg'
        for item in ("url", "filename"):
            if item == "url":
                with pytest.raises(ValueError):
                    assert VLImage.load(url=url)
            else:
                with pytest.raises(FileNotFoundError):
                    assert VLImage.load(filename=filename)

    def test_bad_image_type(self):
        with pytest.raises(LunaSDKException) as ex:
            VLImage(body=b'some text', filename="bytes")
        self.assertLunaVlError(ex, 100033, LunaVLError.InvalidType)

    def test_check_ndarray_type(self):
        image = VLImage.load(filename=ONE_FACE)
        assert image.isValid()
        assert isinstance(image.asNPArray(), np.ndarray)

    def test_convert_color_format_and_check_bgr(self):
        imageFormat = VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.B8G8R8)
        assert imageFormat.isValid()
        R, G, B = np.asarray(Image.open(ONE_FACE)).T
        bgrImageArray = np.array((B, G, R)).T
        assert imageFormat.isBGR() is True
        assert (imageFormat.asNPArray() == bgrImageArray).all()
