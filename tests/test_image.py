from collections import namedtuple
from pathlib import Path

import FaceEngine as fe
import numpy as np
import pytest
from PIL import Image

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage, ColorFormat
from tests.base import BaseTestClass
from tests.resources import ONE_FACE

PATH_TO_IMAGE = Path(ONE_FACE)


class TestImage(BaseTestClass):
    """
    Test of image.
    """
    garbageList = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.garbageList = []

    @classmethod
    def tearDownClass(cls) -> None:
        for path in cls.garbageList:
            Path.unlink(path)

    def test_image_initialize(self):
        binaryBody = PATH_TO_IMAGE.read_bytes()
        bytearrayBody = bytearray(binaryBody)
        imageWithOneFace = Image.open(ONE_FACE)
        npBody = np.asarray(imageWithOneFace)
        coreBody = fe.Image()
        coreBody.load(ONE_FACE)

        InitCase = namedtuple("InitCase", ("initType", "body"))
        cases = (InitCase("bytes", binaryBody), InitCase("numpy array", npBody), InitCase("byte array", bytearrayBody),
                 InitCase("core", coreBody))
        for case in cases:
            with self.subTest(initType=case.initType):
                imageVl = VLImage(body=case.body, filename=case.initType)
                assert imageVl.isValid()
                assert case.initType == imageVl.filename
                assert imageVl.rect == Rect(0, 0, 912, 1080)

    def test_image_load_from_file(self):
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        assert imageWithOneFace.isValid()
        assert imageWithOneFace.rect == Rect(0, 0, 912, 1080)
        assert imageWithOneFace.filename == "one_face.jpg"

    def test_image_rect(self):
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        self.checkRectAttr(imageWithOneFace.rect, isImage=True)

    def test_image_load_from_url(self):
        url = "https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg"
        imageWithOneFace = VLImage.load(url=url)
        assert imageWithOneFace.isValid()
        assert imageWithOneFace.rect == Rect(0, 0, 1000, 1288)
        assert imageWithOneFace.filename == url

    def test_image_bad_filename_or_url(self):
        url = "https://st.kp.yandex.net/images/"
        filename = 'test.jpg'
        for loadType in ("url", "filename"):
            if loadType == "url":
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
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        assert imageWithOneFace.isValid()
        assert (np.asarray(Image.open(ONE_FACE)) == imageWithOneFace.asNPArray()).all()

    def test_convert_color_format_and_check_bgr(self):
        imageFormat = VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.B8G8R8)
        assert imageFormat.isValid()
        assert imageFormat.format.name == "B8G8R8"

        R, G, B = VLImage.load(filename=ONE_FACE).asNPArray().T
        bgrImageArray = np.array((B, G, R)).T
        assert imageFormat.isBGR()
        assert (bgrImageArray == imageFormat.asNPArray()).all()

    def test_unknown_image_format(self):
        with pytest.raises(LunaSDKException) as exceptionInfo:
            VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat("Unknown"))
        self.assertLunaVlError(exceptionInfo, 100029, LunaVLError.InvalidFormat)

    def test_save_image(self):
        for ext in "ppm,jpg,png,tif".split(','):
            pathToTestImage = PATH_TO_IMAGE.parent.joinpath(f"test_image.{ext}")
            VLImage(body=PATH_TO_IMAGE.read_bytes()).save(str(pathToTestImage))
            self.garbageList.append(pathToTestImage)
            VLImage.load(filename=str(pathToTestImage)).isValid()
