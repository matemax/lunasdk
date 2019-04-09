from enum import Enum
from typing import Optional
import requests
import FaceEngine as CoreFE
from numpy import array


class Format(Enum):
    B8G8R8 = 'B8G8R8'
    B8G8R8X8 = 'B8G8R8X8'
    R16 = 'R16'
    R8 = 'R8'
    R8G8B8 = 'R8G8B8'
    R8G8B8X8 = 'R8G8B8X8'
    Unknown = 'Unknown'

    @property
    def coreFormat(self) -> CoreFE.FormatType:
        return getattr(CoreFE.FormatType, self.value)


class VLImage:
    __slots__ = ("_image", "format", "source")

    def __init__(self, body: bytes, imgFormat: Optional[Format] = None):
        if imgFormat is None:
            imgFormat = Format.R8G8B8
        self._image = CoreFE.Image()
        loadResult = self._image.loadFromMemory(body, len(body), imgFormat.coreFormat)
        if loadResult.isError:
            raise ValueError
        self.format = imgFormat
        self.source = body

    @classmethod
    def load(cls, *_, filename: Optional[str] = None, url: Optional[str] = None, npArray: Optional[array] = None,
             imgFormat: Optional[Format] = None) -> 'VLImage':

        """
        Load imag
        Args:
            *_:
            filename:
            url:
            npArray:
            imgFormat:

        Returns:
            vl image

        >>> img = VLImage.load(url='https://st.kp.yandex.net/im/kadr/3/1/4/kinopoisk.ru-Keira-Knightley-3142930.jpg')
        >>> img.rect
        x = 0, y = 0, width = 1000, height = 1288

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
    def rect(self):
        """
        >>> 1+1
        2

        Returns:

        """
        return self._image.getRect()

    def computePitch(self, arg0):  # real signature unknown; restored from __doc__
        """ computePitch(self: FaceEngine.Image, arg0: int) -> int """
        return 0

    @property
    def bitDepth(self):  # real signature unknown; restored from __doc__
        """ getBitDepth(self: FaceEngine.Image) -> int """
        return self._image.getBitDepth()

    @property
    def getByteDepth(self):  # real signature unknown; restored from __doc__
        """ getByteDepth(self: FaceEngine.Image) -> int """
        return self._image.getByteDepth()

    @property
    def channelCount(self):  # real signature unknown; restored from __doc__
        """
        getChannelCount(self: FaceEngine.Image) -> int

        	Returns channel count.
        """
        return self._image.getChannelCount()

    @property
    def channelSize(self):  # real signature unknown; restored from __doc__
        """ getChannelSize(self: FaceEngine.Image) -> int """
        return self._image.getChannelSize()

    @property
    def channelStep(self):  # real signature unknown; restored from __doc__
        """
        getChannelStep(self: FaceEngine.Image) -> int

        	Get channel step.Padding bytes are considered spare channels.
        """
        return self._image.getChannelStep()

    def asNPArray(self):  # real signature unknown; restored from __doc__
        """
        getData(self: FaceEngine.Image) -> array

        	Returns image as numpy array.
        """
        return self._image.getData()

    def isBGR(self):  # real signature unknown; restored from __doc__
        """ isBGR(self: FaceEngine.Image) -> bool """
        return self._image.isBGR()

    def isPadded(self):  # real signature unknown; restored from __doc__
        """ isPadded(self: FaceEngine.Image) -> bool """
        return self._image.isPadded()

    def save(self, *args, **kwargs):  # real signature unknown; restored from __doc__
        """
        save(*args, **kwargs)
        Overloaded function.

        1. save(self: FaceEngine.Image, arg0: str) -> ImageErrorResult

        2. save(self: FaceEngine.Image, arg0: str, arg1: FaceEngine.FormatType) -> ImageErrorResult
        """
        pass


if __name__ == "__main__":
    img = VLImage.load(
        url="https://img.playbuzz.com/image/upload/q_auto:good,f_auto,fl_lossy,w_640,c_limit/v1554130247/xbibkcsaqsx7hpgz4bst.jpg")
    print(img._image.getRect())
    # with open("C:/Users/matem/Desktop/2M4A8223.jpg", "rb") as f:
    #     body = f.read()
    #     img = VLImage(body)
    #     print(1)
