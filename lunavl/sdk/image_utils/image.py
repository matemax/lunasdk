from enum import Enum
from typing import Optional
import requests
import FaceEngine as CoreFE
from numpy import array

try:
    from .geometry import Rect
except ImportError:
    from lunavl.sdk.image_utils.geometry import Rect


class Format(Enum):
    """
    Enum for vl luna image formats
    """
    B8G8R8 = 'B8G8R8'               #: BGR format, 8 byte per pixel
    B8G8R8X8 = 'B8G8R8X8'           #: BGR format with alpha chanel, 8 byte per pixel
    R16 = 'R16'                     #: IR 16
    R8 = 'R8'                       #: IR 8
    R8G8B8 = 'R8G8B8'               #: RGB format, 8 byte per pixel
    R8G8B8X8 = 'R8G8B8X8'           #: RGB format with alpha chanel, 8 byte per pixel
    Unknown = 'Unknown'             #: unknown format

    @property
    def coreFormat(self) -> CoreFE.FormatType:
        """
        Convert  format to luna core format.

        Returns:
            luna core format
        """
        return getattr(CoreFE.FormatType, self.value)

    @staticmethod
    def convertCoreFormat(format: CoreFE.FormatType):
        return getattr(Format, format.name)


class VLImage:
    """
    Class image.

    Attributes:
        _image (CoreFE.Image): core image object
        source (str): source of image (todo change)
        filename (str): filename of the file which is source of image
    """
    __slots__ = ("_image", "source", "filename")

    def __init__(self, body: bytes, imgFormat: Optional[Format] = None, filename: str = ""):
        """
        Init.

        Args:
            body:
            imgFormat:
        """
        if imgFormat is None:
            imgFormat = Format.R8G8B8
        self._image = CoreFE.Image()
        loadResult = self._image.loadFromMemory(body, len(body), imgFormat.coreFormat)
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
            CoreFE.Image().setData(npArray, imgFormat.coreFormat)
        raise ValueError

    @property
    def format(self) -> Format:
        """ getFormat(self: FaceEngine.Image) -> FaceEngine.FormatType

        >>> image = VLImage.load(url='https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg')
        >>> image.format.value
        'R8G8B8'
        """
        return Format.convertCoreFormat(self._image.getFormat())

    @property
    def rect(self) -> Rect:
        """
        Get rect of image.

        Returns:
            rect of the image
        """
        return Rect.fromCoreRect(self._image.getRect())

    def computePitch(self, arg0):
        """
        todo: description and typing
        Args:
            arg0:

        Returns:

        """
        return self._image.computePitch()

    @property
    def bitDepth(self) -> int:
        """

        Returns:

        """
        return self._image.getBitDepth()

    @property
    def getByteDepth(self):  # real signature unknown; restored from __doc__
        """ getByteDepth(self: FaceEngine.Image) -> int """
        return self._image.getByteDepth()

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
        return self._image.getChannelCount()

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
        return self._image.getChannelSize()

    @property
    def channelStep(self) -> int:
        """
        Get chanel step.
        todo: more description

        Returns:
            channel size in bytes.

        >>> img = VLImage.load(url='https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg')
        >>> img.channelStep
        3
        """
        return self._image.getChannelStep()

    def asNPArray(self) -> array:
        """
        Get image as numpy array.

        Returns:
            numpy array
        todo: doctest
        """
        return self._image.getData()

    def isBGR(self) -> bool:
        """
        Check whether format image is bgr or not.

        Returns:
            True if the image is bgr image otherwise False

        >>> VLImage.load(url='https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg').isBGR()
        False
        """
        return self._image.isBGR()

    def isPadded(self) -> bool:
        """
        todo: more description
        Returns:

        """
        return self._image.isPadded()

    def save(self, *args, **kwargs):  # real signature unknown; restored from __doc__
        """
        todo: do it
        """
        pass
