"""
Module realize VLImage - structure for storing image in special format.
"""
from enum import Enum
from pathlib import Path
from typing import Optional, Union
import requests
from FaceEngine import FormatType, Image as CoreImage  # pylint: disable=E0611,E0401
from numpy import ndarray

from .geometry import Rect


class ImageFormat(Enum):
    """
    Enum for image format
    """
    #: jpg
    JPG = 'jpg'
    #: png
    PNG = 'png'
    #: ppm
    PPM = 'ppm'
    #: tif
    TIF = 'tif'


class ColorFormat(Enum):
    """
    Enum for vl luna color formats
    """
    #: 3 channel, 8 bit per channel, B-G-R color order format;
    B8G8R8 = 'B8G8R8'
    #: 3 channel, 8 bit per channel, B-G-R color order format with 8 bit padding before next pixel;
    B8G8R8X8 = 'B8G8R8X8'
    #: 1 channel, 8 bit per channel format;
    R16 = 'R16'
    #: 1 channel, 8 bit per channel format;
    R8 = 'R8'
    #: 3 channel, 8 bit per channel, R-G-B color order format;
    R8G8B8 = 'R8G8B8'
    #: 3 channel, 8 bit per channel, R-G-B color order format with 8 bit padding before next pixel;
    R8G8B8X8 = 'R8G8B8X8'
    #: unknown format
    Unknown = 'Unknown'

    @property
    def coreFormat(self) -> FormatType:
        """
        Convert  format to luna core format.

        Returns:
            luna core format
        """
        return getattr(FormatType, self.value)

    @staticmethod
    def convertCoreFormat(imageFormat: FormatType) -> 'ColorFormat':
        """
        Convert FormatType to Format

        Args:
            imageFormat: core image format

        Returns:
            corresponding lunavl image format

        """
        return getattr(ColorFormat, imageFormat.name)


class VLImage:
    """
    Class image.

    Attributes:
        coreImage (CoreFE.Image): core image object
        source (str): source of image (todo change)
        filename (str): filename of the file which is source of image
    """
    __slots__ = ("coreImage", "source", "filename")

    def __init__(self, body: Union[bytes, ndarray, CoreImage], imgFormat: Optional[ColorFormat] = None,
                 filename: str = ""):
        """
        Init.

        Args:
            body: body of image - bytes numpy array or core image
            imgFormat: img format
            filename: user mark a source of image
        """
        if imgFormat is None:
            imgFormat = ColorFormat.R8G8B8
        self.coreImage = CoreImage()

        if isinstance(body, CoreImage):
            self.coreImage = body
        elif isinstance(body, bytes):
            loadResult = self.coreImage.loadFromMemory(body, len(body), imgFormat.coreFormat)
            if loadResult.isError:
                #: todo: raise correct error.
                raise ValueError
        elif isinstance(body, ndarray):
            #: todo, format ?????
            self.coreImage.setData(body, imgFormat.coreFormat)
        else:
            raise TypeError("wtf  image type")

        self.source = body
        self.filename = filename

    @classmethod
    def load(cls, *_, filename: Optional[str] = None, url: Optional[str] = None,
             imgFormat: Optional[ColorFormat] = None) -> 'VLImage':

        """
        Load imag from numpy array or file or url.

        Args:
            *_: for remove positional argument
            filename: filename
            url: url
            imgFormat:

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
                img = cls(body, imgFormat)
                img.source = path.name
                return img

        if url is not None:
            response = requests.get(url=url)
            if response.status_code == 200:
                img = cls(response.content, imgFormat)
                img.source = url
                return img
        raise ValueError

    @property
    def format(self) -> ColorFormat:
        """ getFormat(self: FaceEngine.Image) -> FaceEngine.FormatType

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

    def asNPArray(self) -> ndarray:
        """
        Get image as numpy array.

        Returns:
            numpy array
        todo: doctest
        """
        return self.coreImage.getData()

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

    def save(self, filename: str):
        """
        Save image to disk. Support image format: *ppm, jpg, png, tif*.

        Args:
            filename: filename
        Raises:
            todo it
        """
        saveRes = self.coreImage.save(filename)
        if saveRes.isError:
            raise ValueError

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
