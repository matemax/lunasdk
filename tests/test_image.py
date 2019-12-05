import os
from collections import namedtuple
from pathlib import Path

import FaceEngine as fe
import numpy as np
import pytest
from PIL import Image

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage, ColorFormat, ImageFormat
from tests.base import BaseTestClass
from tests.resources import ONE_FACE

SINGLE_CHANNEL_IMAGE = np.asarray(Image.open(ONE_FACE).convert("L"))
ONE_16_BIT_IMAGE = np.asarray(Image.open(ONE_FACE).convert("L"), dtype=np.uint16)


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
        """
        Test create VLImage with image body
        """
        binaryBody = Path(ONE_FACE).read_bytes()
        bytearrayBody = bytearray(binaryBody)
        imageWithOneFace = Image.open(ONE_FACE)
        npBody = np.asarray(imageWithOneFace)
        coreBody = fe.Image()
        coreBody.load(ONE_FACE)

        InitCase = namedtuple("InitCase", ("initType", "body"))
        cases = (
            InitCase("bytes", binaryBody),
            InitCase("numpy array", npBody),
            InitCase("byte array", bytearrayBody),
            InitCase("core", coreBody),
        )
        for case in cases:
            with self.subTest(initType=case.initType):
                imageVl = VLImage(body=case.body, filename=case.initType)
                assert imageVl.isValid()
                assert case.initType == imageVl.filename
                assert imageVl.rect == Rect(0, 0, 912, 1080)

    def test_load_image_from_file(self):
        """
        Test load image from file
        """
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        assert imageWithOneFace.isValid()
        assert imageWithOneFace.rect == Rect(0, 0, 912, 1080)
        assert imageWithOneFace.filename == "one_face.jpg"

    def test_image_rect(self):
        """
        Test validate image rect
        """
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        self.checkRectAttr(imageWithOneFace.rect)

    def test_load_image_from_url(self):
        """
        Test load image from url
        """
        url = "https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg"
        imageWithOneFace = VLImage.load(url=url)
        assert imageWithOneFace.isValid()
        assert imageWithOneFace.rect == Rect(0, 0, 1000, 1288)
        assert imageWithOneFace.filename == url

    def test_not_set_image_filename_or_url(self):
        """
        Test check load image if filename or url is not set
        """
        for loadType in ("url", "filename"):
            with self.subTest(loadType=loadType):
                if loadType == "url":
                    with pytest.raises(ValueError):
                        assert VLImage.load(url=None)
                else:
                    with pytest.raises(ValueError):
                        assert VLImage.load(filename=None)

    def test_invalid_image_type(self):
        """
        Test invalid image type
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            VLImage(body=b"some text", filename="bytes")
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidType)

    def test_check_ndarray_type(self):
        """
        Test check numpy array conversion
        """
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        assert imageWithOneFace.isValid()
        assert (np.asarray(Image.open(ONE_FACE)) == imageWithOneFace.asNPArray()).all()

    def test_convert_color_format(self):
        """
        Test check color format conversion
        """
        colorImage = VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.B8G8R8)
        assert colorImage.isValid()
        assert colorImage.format == ColorFormat.B8G8R8

        R, G, B = VLImage.load(filename=ONE_FACE).asNPArray().T
        bgrImageArray = np.array((B, G, R)).T
        assert colorImage.isBGR()
        assert (bgrImageArray == colorImage.asNPArray()).all()

    def test_unknown_image_format(self):
        """
        Test check load image if color format is unknown
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat("Unknown"))
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidFormat)

    def test_save_image(self):
        """
        Test save image to directory
        """
        for ext in ImageFormat:
            pathToTestImage = Path(ONE_FACE).parent.joinpath(f"image_test.{ext.value}")
            VLImage(body=Path(ONE_FACE).read_bytes()).save(str(pathToTestImage))
            self.garbageList.append(pathToTestImage)
            VLImage.load(filename=str(pathToTestImage)).isValid()

    def test_image_format_padded(self):
        """
        Test check image format has padded bytes
        """
        assert VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.R8G8B8X8).isPadded()
        assert VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.B8G8R8X8).isPadded()

    def test_image_format_not_padded(self):
        """
        Test check image format with no padded bytes
        """
        assert VLImage(body=SINGLE_CHANNEL_IMAGE, imgFormat=ColorFormat.R16).isPadded() is False
        assert VLImage(body=SINGLE_CHANNEL_IMAGE, imgFormat=ColorFormat.R8).isPadded() is False
        assert VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.R8G8B8).isPadded() is False
        assert VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.B8G8R8).isPadded() is False

    def test_invalid_image_conversion(self):
        """
        Test convert image to one channel format
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            VLImage.load(filename=ONE_FACE, imgFormat=ColorFormat.R16)
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidConversion)

    def test_invalid_image_data_size(self):
        """
        Test invalid image data size
        """
        for body in (b"", bytearray()):
            with self.subTest(body=body):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    VLImage(body=body, filename="bytes")
                self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidDataSize)

    def test_bad_image_type(self):
        """
        Test bad image body type
        """
        with pytest.raises(TypeError):
            VLImage(body=VLImage.coreImage, filename="coreImage")

    @pytest.mark.skip("core bug")
    def test_failed_to_save_jpeg_format(self):
        """
        Test saving single channel image with different color format
        """
        for color in [ColorFormat.R16, ColorFormat.R8G8B8X8, ColorFormat.B8G8R8X8]:
            with self.subTest(colorFormat=color):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    VLImage(body=SINGLE_CHANNEL_IMAGE, imgFormat=color).save(f"test_jpeg.jpg")
                self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidBitmap)
