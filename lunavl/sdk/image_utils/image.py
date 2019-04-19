"""
Module realize VLImage - structure for storing image in special format.
"""
from enum import Enum
from typing import Optional
import requests
from FaceEngine import FormatType, Image as CoreImage  # pylint: disable=E0611,E0401
from numpy import array

from .geometry import Rect


class Format(Enum):
    """
    Enum for vl luna image formats
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
    def convertCoreFormat(imageFormat: FormatType) -> 'Format':
        """
        Convert FormatType to Format

        Args:
            imageFormat: core image format

        Returns:
            corresponding lunavl image format

        """
        return getattr(Format, imageFormat.name)


class VLImage:
    """
    Class image.

    Attributes:
        coreImage (CoreFE.Image): core image object
        source (str): source of image (todo change)
        filename (str): filename of the file which is source of image
    """
    __slots__ = ("coreImage", "source", "filename")

    def __init__(self, body: bytes, imgFormat: Optional[Format] = None, filename: str = ""):
        """
        Init.

        Args:
            body:
            imgFormat:
        """
        if imgFormat is None:
            imgFormat = Format.R8G8B8
        self.coreImage = CoreImage()
        loadResult = self.coreImage.loadFromMemory(body, len(body), imgFormat.coreFormat)
        if loadResult.isError:
            #: todo: raise correct error.
            raise ValueError
        self.source = body
        self.filename = filename

    @classmethod
    def load(cls, *_, filename: Optional[str] = None, url: Optional[str] = None, npArray: Optional[array] = None,
             imgFormat: Optional[Format] = None) -> 'VLImage':

        """
        Load imag from numpy array or file or url.

        Args:
            *_: for remove positional argument
            filename: filename
            url: url
            npArray:
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
            with open(filename, "rb") as file:
                body = file.read()
                img = cls(body, imgFormat)
                img.source = filename
                return img

        if url is not None:
            response = requests.get(url=url)
            if response.status_code == 200:
                img = cls(response.content, imgFormat)
                img.source = url
                return img
        if npArray is not None:
            CoreImage().setData(npArray, imgFormat.coreFormat)
        raise ValueError

    @property
    def format(self) -> Format:
        """ getFormat(self: FaceEngine.Image) -> FaceEngine.FormatType

        >>> image = VLImage.load(url='https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg')
        >>> image.format.value
        'R8G8B8'
        """
        return Format.convertCoreFormat(self.coreImage.getFormat())

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

    def asNPArray(self) -> array:
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

    def save(self, *args, **kwargs):  # real signature unknown; restored from __doc__
        """
        todo: do it
        """
        pass
