import os

from collections import namedtuple
from pathlib import Path
from typing import List, Union

from lunavl.sdk.errors.errors import LunaVLError
import FaceEngine as fe
import numpy as np
import pytest
from PIL import Image


from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.image_utils.geometry import Rect
from lunavl.sdk.image_utils.image import VLImage, ColorFormat, ImageFormat
from tests.base import BaseTestClass
from tests.resources import ONE_FACE

SINGLE_CHANNEL_IMAGE: Image.Image = Image.open(ONE_FACE).convert("L")
IMAGE = Image.open(ONE_FACE)
RESTRICTED_COLOR_FORMATS = {ColorFormat.R16, ColorFormat.Unknown}


class TestImage(BaseTestClass):
    """
    Test of image.
    """

    #: list with paths to test images
    filesToDelete: List[Union[Path, str]]

    @classmethod
    def setup_class(cls) -> None:
        super().setup_class()
        cls.filesToDelete = []

    @classmethod
    def teardown_class(cls) -> None:
        super().teardown_class()
        for path in cls.filesToDelete:
            if isinstance(path, Path):
                Path.unlink(path)
            elif isinstance(path, str):
                os.remove(path)
            else:
                raise NotImplementedError(f"{type(path)}")

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
            InitCase("numpy array", np.asarray(imageWithOneFace)),
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
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidType.format("Unsupported type"))

    def test_check_ndarray_type(self):
        """
        Test check numpy array conversion
        """
        image = Image.open(ONE_FACE)
        imageWithOneFace = VLImage(image)
        assert (np.asarray(image) == imageWithOneFace.asNPArray()).all()

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
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidFormat.format("Unsupported format"))

    def test_save_image(self):
        """
        Test save image to directory and check format
        """
        for ext in ImageFormat:
            with self.subTest(extension=ext):
                pathToTestImage = Path(ONE_FACE).parent.joinpath(f"image_test.{ext.value}")
                VLImage(body=Path(ONE_FACE).read_bytes()).save(pathToTestImage.as_posix())
                self.filesToDelete.append(pathToTestImage)

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
                self.assertLunaVlError(
                    exceptionInfo, LunaVLError.InvalidConversion.format("Required conversion not implemented")
                )

    def test_invalid_image_data_size(self):
        """
        Test invalid image data size
        """
        for body in (b"", bytearray()):
            with self.subTest(body=body):
                with pytest.raises(LunaSDKException) as exceptionInfo:
                    VLImage(body=body, filename="bytes")
                self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidDataSize.format("Bad input data size"))

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
        self.filesToDelete.append(IMG_PATH)
        allColorFormats = set(ColorFormat) - RESTRICTED_COLOR_FORMATS
        failColorFormats = {ColorFormat.B8G8R8X8, ColorFormat.R8G8B8X8}
        for color in allColorFormats:
            with self.subTest(colorFormat=color):
                if color not in failColorFormats:
                    VLImage(body=IMAGE).save(IMG_PATH, colorFormat=color)
                else:
                    with pytest.raises(LunaSDKException) as exceptionInfo:
                        VLImage(body=IMAGE).save(IMG_PATH, colorFormat=color)
                    self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidBitmap.format("Bitmap error"))

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

    def test_from_numpy_array(self):
        """
        Test init image from different formats with validation
        """
        colorToNdArrayMap = self.generateColorToArrayMap()
        for color, ndarray in colorToNdArrayMap.items():
            with self.subTest(color=color.name):
                img = VLImage.fromNumpyArray(ndarray, color)
                assert color == img.format, img.format

    def test_convert(self):
        """
        Test convert image combinations (every to every).
        """
        allImages = self.getColorToImageMap()
        unsupportedConversions = (
            (ColorFormat.IR_X8X8X8, ColorFormat.R16),
            (ColorFormat.R8, ColorFormat.R16),
            (ColorFormat.B8G8R8, ColorFormat.R16),
            (ColorFormat.R8G8B8, ColorFormat.R16),
            (ColorFormat.B8G8R8X8, ColorFormat.R16),
            (ColorFormat.R8G8B8X8, ColorFormat.R16),
            (ColorFormat.R16, ColorFormat.IR_X8X8X8),
            (ColorFormat.R16, ColorFormat.R8),
            (ColorFormat.R16, ColorFormat.B8G8R8),
            (ColorFormat.R16, ColorFormat.R8G8B8),
            (ColorFormat.R16, ColorFormat.B8G8R8X8),
            (ColorFormat.R16, ColorFormat.R8G8B8X8),
        )

        for source in set(allImages):
            sourceImage = allImages[source]
            for target in set(allImages):
                with self.subTest(source=source.name, target=target.name):
                    if (source, target) in unsupportedConversions:
                        with pytest.raises(LunaSDKException) as exceptionInfo:
                            sourceImage.convert(target)
                        self.assertLunaVlError(
                            exceptionInfo, LunaVLError.InvalidConversion.format("Required conversion not implemented")
                        )
                    else:
                        targetImg = sourceImage.convert(target)
                        assert target == targetImg.format, targetImg.format
