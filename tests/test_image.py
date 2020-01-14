import os
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Union

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

SINGLE_CHANNEL_IMAGE = Image.open(ONE_FACE).convert("L")
IMAGE = Image.open(ONE_FACE)
RESTRICTED_COLOR_FORMATS = {ColorFormat.R16, ColorFormat.Unknown}


class TestImage(BaseTestClass):
    """
    Test of image.
    """

    #: list with paths to test images
    garbageImagesList: List[Union[Path, str]]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.garbageImagesList = []

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        for path in cls.garbageImagesList:
            if isinstance(path, Path):
                Path.unlink(path)
            elif isinstance(path, str):
                os.remove(path)
            else:
                raise NotImplemented(f"{type(path)}")

    @staticmethod
    def getColorToImageMap() -> Dict[ColorFormat, VLImage]:
        """
        Get all available images.

        Returns:
            color format to vl image map
        """
        restrictedColorFormats = {ColorFormat.R16, ColorFormat.Unknown}
        allColorFormats = set(ColorFormat) - restrictedColorFormats
        baseImage = VLImage(IMAGE)

        failColorFormats = {ColorFormat.R8G8B8X8, ColorFormat.B8G8R8X8}
        R, G, B = baseImage.asNPArray().T
        X = np.ndarray(B.shape)
        bgrxImage = VLImage.fromNumpyArray(np.array((B, G, R, X)).T, ColorFormat.B8G8R8X8)

        allImages = {
            colorFormat: VLImage(IMAGE.convert("RGBX"))
            if colorFormat == ColorFormat.R8G8B8X8
            else bgrxImage
            if colorFormat == ColorFormat.B8G8R8X8
            else VLImage(baseImage.coreImage, colorFormat=colorFormat)
            for colorFormat in allColorFormats
        }
        assert None not in allImages.values(), f"Unsupported type"
        return allImages

    def test_image_initialize(self):
        """
        Test create VLImage with image body
        """
        binaryBody = Path(ONE_FACE).read_bytes()
        bytearrayBody = bytearray(binaryBody)
        imageWithOneFace = Image.open(ONE_FACE)
        coreBody = fe.Image()
        coreBody.load(ONE_FACE)

        InitCase = namedtuple("InitCase", ("initType", "body"))
        cases = (
            InitCase("bytes", binaryBody),
            InitCase("byte array", bytearrayBody),
            InitCase("core", coreBody),
            InitCase("pillow img", imageWithOneFace),
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
        colorImage = VLImage.load(filename=ONE_FACE, colorFormat=ColorFormat.B8G8R8)
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
            VLImage.load(filename=ONE_FACE, colorFormat=ColorFormat("Unknown"))
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidFormat)

    def test_save_image(self):
        """
        Test save image to directory and check format
        """
        for ext in ImageFormat:
            with self.subTest(extension=ext):
                pathToTestImage = Path(ONE_FACE).parent.joinpath(f"image_test.{ext.value}")
                VLImage(body=Path(ONE_FACE).read_bytes()).save(pathToTestImage.as_posix())
                self.garbageImagesList.append(pathToTestImage)

                VLImage.load(filename=pathToTestImage.as_posix()).isValid()
                pillowImage = Image.open(pathToTestImage.as_posix())
                if pillowImage.verify() is None:
                    assert pillowImage.format == ext.name
                else:
                    raise TypeError("Invalid Image")

    def test_image_format_padding(self):
        """
        Test check padded bytes in all image format
        """
        paddedFormats = (ColorFormat.R8G8B8X8, ColorFormat.B8G8R8X8)

        for colorFormat, image in self.getColorToImageMap().items():
            with self.subTest(colorFormat=colorFormat.name):
                assert (colorFormat in paddedFormats) == image.isPadded()

    def test_invalid_image_conversion(self):
        """
        Test convert image to one channel format
        """
        for colorFormat in RESTRICTED_COLOR_FORMATS - {ColorFormat.Unknown}:
            with self.subTest(colorFormat=colorFormat):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    VLImage.load(filename=ONE_FACE, colorFormat=colorFormat)
                self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidConversion)

    def test_invalid_image_data_size(self):
        """
        Test invalid image data size
        """
        for body in (b"", bytearray(), b"1234", b"JPEG"):
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

    def test_save_jpeg_in_all_formats(self):
        """
        Test saving single channel image with different color format
        """
        IMG_PATH = os.path.abspath("test_jpeg.jpg")
        self.garbageImagesList.append(IMG_PATH)
        allColorFormats = set(ColorFormat) - RESTRICTED_COLOR_FORMATS
        failColorFormats = {ColorFormat.B8G8R8X8, ColorFormat.R8G8B8X8}
        for color in allColorFormats:
            with self.subTest(colorFormat=color):
                if color not in failColorFormats:
                    VLImage(body=IMAGE).save(IMG_PATH, colorFormat=color)
                    im = Image.open(IMG_PATH)
                    im.load()
                    if color != ColorFormat.B8G8R8:
                        self.assertEqual(color, ColorFormat.load(im.mode))
                else:
                    with pytest.raises(LunaSDKException) as exceptionInfo:
                        VLImage(body=IMAGE).save(IMG_PATH, colorFormat=color)
                    self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidBitmap)

    def test_zero_numpy_array(self):
        """
        Test image validation with a zero array:
            (0, 0, 0), (0, 0, 0), (0, 0, 0)
            (0, 0, 0), (0, 0, 0), (0, 0, 0)
            (0, 0, 0), (0, 0, 0), (0, 0, 0)
        """
        zeroArray = np.zeros(shape=(3, 3, 3))
        blackImage = VLImage.fromNumpyArray(arr=zeroArray, inputColorFormat="RGB", filename="array")
        assert blackImage.isValid()
        self.checkRectAttr(blackImage.rect)

    def test_convert(self):
        """
        Test convert image combinations (every to every).
        """
        allImages = self.getColorToImageMap()
        for source in allImages:
            sourceImage = allImages[source]
            for target in allImages:
                with self.subTest(source=source.name, target=target.name):
                    sourceImage.convert(target)
