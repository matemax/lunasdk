"""
Module realize VLImage - structure for storing image in special format.
"""
from copy import copy
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
import requests
from FaceEngine import FormatType, Image as CoreImage  # pylint: disable=E0611,E0401
from PIL import Image as pilImage
from PIL.Image import Image as PilImage

from .geometry import Rect
from .pil.np import getNPImageType, pilToNumpy
from ..errors.exceptions import assertError


class ImageFormat(Enum):
    """
    Enum for image format
    """

    #: jpg
    JPEG = "jpg"
    #: png
    PNG = "png"
    #: ppm
    PPM = "ppm"
    #: tif
    TIFF = "tif"
    #: bmp
    BMP = "bmp"


class RotationAngle(Enum):
    """
    Enum for rotation angle
    """

    #: zero rotation angle
    ANGLE_0 = "Normal"
    #: rotate counter-clockwise
    ANGLE_90 = "Left"
    #:  clockwise
    ANGLE_270 = "Right"
    #: rotate upside-down
    ANGLE_180 = "UpsideDown"


class ColorFormat(Enum):
    """
    Enum for vl luna color formats
    """

    #: 3 channel, 8 bit per channel, B-G-R color order format;
    B8G8R8 = "B8G8R8"
    #: 3 channel, 8 bit per channel, B-G-R color order format with 8 bit padding before next pixel;
    B8G8R8X8 = "B8G8R8X8"
    #: 3 channel, 8 bit per channel format with InfraRed semantics
    IR_X8X8X8 = "IR_X8X8X8"
    #: 1 channel, 16 bit per channel format;
    R16 = "R16"
    #: 1 channel, 8 bit per channel format;
    R8 = "R8"
    #: 3 channel, 8 bit per channel, R-G-B color order format;
    R8G8B8 = "R8G8B8"
    #: 3 channel, 8 bit per channel, R-G-B color order format with 8 bit padding before next pixel;
    R8G8B8X8 = "R8G8B8X8"
    #: unknown format
    Unknown = "Unknown"

    @property
    def coreFormat(self) -> FormatType:
        """
        Convert  format to luna core format.

        Returns:
            luna core format
        """
        return getattr(FormatType, self.value)

    @staticmethod
    def convertCoreFormat(imageFormat: FormatType) -> "ColorFormat":
        """
        Convert FormatType to Format

        Args:
            imageFormat: core image format

        Returns:
            corresponding lunavl image format

        """
        return getattr(ColorFormat, imageFormat.name)

    @classmethod
    def load(cls, colorFormat: str) -> "ColorFormat":
        """
        Load color format from known sources:
            1. some PIL image "mode" https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
            2. FSDK color format

        Args:
            colorFormat: input color format

        Returns:
            corresponding lunavl image format

        Raises:
            NotImplementedError if color format is not supported
        """

        if colorFormat == "RGB":
            return cls.R8G8B8
        if colorFormat == "RGBa":
            return cls.R8G8B8X8
        if colorFormat == "RGBA":
            return cls.R8G8B8X8
        if colorFormat == "RGBX":
            return cls.R8G8B8X8
        if colorFormat == "BGR":
            return cls.B8G8R8
        if colorFormat == "BGRa":
            return cls.B8G8R8X8
        if colorFormat == "BGRx":
            return cls.B8G8R8X8
        if colorFormat == "RGB":
            return cls.R8G8B8
        if colorFormat in "LP":
            return cls.R8

        try:
            return getattr(cls, colorFormat)
        except AttributeError:
            pass

        raise ValueError(f"Cannot load '{colorFormat}' color format.")


class VLImage:
    """
    Class image.

    Attributes:
        coreImage (CoreFE.Image): core image object
        source (Union[bytes, bytearray, PilImage, CoreImage]): body of image
        filename (str): filename of the file which is source of image
    """

    __slots__ = ("coreImage", "source", "filename")

    def __init__(
        self,
        body: Union[bytes, bytearray, PilImage, CoreImage],
        colorFormat: Optional[ColorFormat] = None,
        filename: str = "",
    ):
        """
        Init.

        Args:
            body: body of image - bytes numpy array or core image
            colorFormat: img format to cast into
            filename: user mark a source of image
        Raises:
            TypeError: if body has incorrect type
            LunaSDKException: if failed to load image to sdk Image
        """
        if isinstance(body, bytearray):
            body = bytes(body)

        if isinstance(body, CoreImage):
            if colorFormat is None or colorFormat.coreFormat == body.getFormat():
                self.coreImage = body
            else:
                error, self.coreImage = body.convert(colorFormat.coreFormat)
                assertError(error)

        elif isinstance(body, bytes):
            self.coreImage = CoreImage()
            imgFormat = (colorFormat or ColorFormat.R8G8B8).coreFormat
            error = self.coreImage.loadFromMemory(body, len(body), imgFormat)
            assertError(error)

        elif isinstance(body, np.ndarray):
            mode = getNPImageType(body)
            self.coreImage = self._coreImageFromNumpyArray(
                ndarray=body, inputColorFormat=ColorFormat.load(mode), colorFormat=colorFormat or ColorFormat.R8G8B8
            )
        elif isinstance(body, PilImage):
            array = pilToNumpy(body)
            # prevent palette-mode colorful images conversion to grayscale image by converting it
            # with considering palette info
            # modes description: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
            blackNWhiteModes = ("1", "L", "LA", "La")
            if body.mode == "P" and body.palette is not None and body.palette.mode not in blackNWhiteModes:
                body = body.convert()
            inputColorFormat = ColorFormat.load(body.mode)
            self.coreImage = self._coreImageFromNumpyArray(
                ndarray=array, inputColorFormat=inputColorFormat, colorFormat=colorFormat or ColorFormat.R8G8B8
            )
        else:
            raise TypeError(f"Bad image type: {type(body)}")

        self.source = body
        self.filename = filename

    @classmethod
    def rotate(cls, image: "VLImage", angle: RotationAngle):
        """
        Get rotated copy of VLImage
        Args:
            image: vl image
            angle: angle for rotation
        Warnings:
            remove after FSDK-2809
        Returns:
            rotated vl image
        """
        if angle == RotationAngle.ANGLE_90:
            angleForRotation = pilImage.ROTATE_90
        elif angle == RotationAngle.ANGLE_270:
            angleForRotation = pilImage.ROTATE_270
        elif angle == RotationAngle.ANGLE_180:
            angleForRotation = pilImage.ROTATE_180
        else:
            return copy(image)

        newPilImage = image.asPillow().transpose(angleForRotation)

        return cls(newPilImage, filename=image.filename)

    @classmethod
    def load(
        cls, *, filename: Optional[str] = None, url: Optional[str] = None, colorFormat: Optional[ColorFormat] = None
    ) -> "VLImage":

        """
        Load image from numpy array or file or url.

        Args:
            *: for positional argument removal
            filename: filename
            url: url
            colorFormat: img format to cast into

        Returns:
            vl image
        Raises:
            ValueError: if no one argument  did not set.

        >>> VLImage.load(url='https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg').rect
        x = 0, y = 0, width = 1000, height = 1288

        todo: more doc test
        """
        if filename is not None:
            path = Path(filename)
            with path.open("rb") as file:
                body = file.read()
                img = cls(body, colorFormat)
                img.filename = path.name
                return img

        if url is not None:
            response = requests.get(url=url)
            if response.status_code == 200:
                img = cls(response.content, colorFormat)
                img.filename = url
                return img
        raise ValueError

    @staticmethod
    def _coreImageFromNumpyArray(
        ndarray: np.ndarray, inputColorFormat: ColorFormat, colorFormat: Optional[ColorFormat] = None
    ) -> CoreImage:
        """
        Load VLImage from numpy array into `self`.

        Args:
            ndarray: numpy pixel array
            inputColorFormat: numpy pixel array format
            colorFormat: pixel format to cast into

        Returns:
            core image instance
        """
        baseCoreImage = CoreImage()
        baseCoreImage.setData(ndarray, inputColorFormat.coreFormat)
        if colorFormat is None or baseCoreImage.getFormat() == colorFormat.coreFormat:
            return baseCoreImage

        error, convertedCoreImage = baseCoreImage.convert(colorFormat.coreFormat)
        assertError(error)

        return convertedCoreImage

    @classmethod
    def fromNumpyArray(
        cls,
        arr: np.ndarray,
        inputColorFormat: Optional[Union[str, ColorFormat]] = None,
        colorFormat: Optional[ColorFormat] = None,
        filename: str = "",
    ) -> "VLImage":
        """
        Load VLImage from numpy array.

        Args:
            arr: numpy pixel array
            inputColorFormat: input numpy pixel array format
            colorFormat: pixel format to cast into
            filename: optional filename

        Returns:
            vl image
        """
        if isinstance(inputColorFormat, str):
            inputColorFormat = ColorFormat.load(inputColorFormat)

        if inputColorFormat is None:
            imageType = getNPImageType(arr)
            inputColorFormat = ColorFormat.load(imageType)

        coreImage = cls._coreImageFromNumpyArray(
            ndarray=arr, inputColorFormat=inputColorFormat, colorFormat=colorFormat
        )
        img = cls(coreImage, filename=filename)
        img.source = arr
        return img

    @property
    def format(self) -> ColorFormat:
        """
        Get image format.

        >>> image = VLImage.load(url='https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg')
        >>> image.format.value
        'R8G8B8'
        """
        return ColorFormat.convertCoreFormat(self.coreImage.getFormat())

    @property
    def rect(self) -> Rect:
        """
        Get rect of image.

        Returns:
            rect of the image
        """
        return Rect.fromCoreRect(self.coreImage.getRect())

    def computePitch(self, rowWidth) -> int:
        """
        Compute row size in bytes

        Args:
            rowWidth: row width in pixels.

        Returns:
            row size in bytes.
        """
        return self.coreImage.computePitch(rowWidth)

    @property
    def bitDepth(self) -> int:
        """
        Get number of bits per pixel.

        Returns:
            number of bits per pixel.
        """
        return self.coreImage.getBitDepth()

    @property
    def getByteDepth(self) -> int:
        """
        Get number of bytes per pixel.

        Returns:
            number of bytes per pixel.
        """
        return self.coreImage.getByteDepth()

    @property
    def channelCount(self) -> int:
        """
        Get chanel count of the image.

        Returns:
            channel count.

        >>> img = VLImage.load(url='https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg')
        >>> img.channelCount
        3
        """
        return self.coreImage.getChannelCount()

    @property
    def channelSize(self) -> int:
        """
        Get size of one chanel in bites.

        Returns:
            channel size in bytes.

        >>> img = VLImage.load(url='https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg')
        >>> img.channelSize
        8
        """
        return self.coreImage.getChannelSize()

    @property
    def channelStep(self) -> int:
        """
        Get chanel step.
        todo: more description

        Returns:
            channel size in bytes.
        Notes:
            padding bytes are considered spare channels.

        >>> img = VLImage.load(url='https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg')
        >>> img.channelStep
        3
        """
        return self.coreImage.getChannelStep()

    def asNPArray(self) -> np.ndarray:
        """
        Get image as numpy array.

        !!!WARNING!!! Does NOT return the same image as in the self.coreImage.

        Returns:
            numpy array
        todo: doctest
        """
        if self.format == ColorFormat.R16:
            return self.coreImage.getDataR16()
        return self.coreImage.getData()

    def asPillow(self) -> PilImage:
        """
        Get image as pillow image.

        !!!WARNING!!! Does NOT return the same image as in the self.coreImage.

        Returns:
            pillow image
        todo: doctest
        """
        imageArray = self.asNPArray()
        return pilImage.fromarray(imageArray)

    def isBGR(self) -> bool:
        """
        Check whether format image is bgr or not.

        Returns:
            True if the image is bgr image otherwise False
        Notes:
            padding is ignored for padded channels.

        >>> VLImage.load(url='https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg').isBGR()
        False
        """
        return self.coreImage.isBGR()

    def isPadded(self) -> bool:
        """
        Determinate image format has padding bytes or not.

        Returns:
            true if image format has padding bytes.
        todo examples
        """
        return self.coreImage.isPadded()

    def save(self, filename: str, colorFormat: Optional[ColorFormat] = None):
        """
        Save image to disk. Support image format: *ppm, jpg, png, tif*.

        Args:
            filename: filename
            colorFormat: color format to save image in
        Raises:
            LunaSDKException: if failed to save image to sdk Image
        """
        if colorFormat is None:
            error = self.coreImage.save(filename)
        else:
            error = self.coreImage.save(filename, colorFormat.coreFormat)
        assertError(error)

    def convertToBinaryImg(self, imageFormat: ImageFormat = ImageFormat.PPM) -> bytes:
        """
        Convert VL image to binary image
        Args:
            imageFormat: format
        Returns:
            bytes
        """
        pass

    def isValid(self) -> bool:
        """
        Check image is valid loaded  to core image or not

        Returns:
            True if image is valid otherwise False
        """
        return self.coreImage.isValid()

    def convert(self, colorFormat: ColorFormat) -> "VLImage":
        """
        Convert current VLImage into image with another color format.

        Args:
            colorFormat: color format to convert into

        Returns:
            converted vl image

        Raises:
            LunaSDKException: if failed to convert image
        """
        error, coreImage = self.coreImage.convert(colorFormat.coreFormat)
        assertError(error)

        return self.__class__(body=coreImage, filename=self.filename)
