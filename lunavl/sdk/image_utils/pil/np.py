"""
Module contains helper functions for a pillow image conversion into np array
"""

import PIL.Image
import numpy as np
from PIL.Image import Image, _fromarray_typemap as imageTypeMap


def getNPImageType(arr: np.ndarray) -> str:
    """
    Get numpy image type
    Args:
        arr: numpy array
    Returns:
        image type which pillow associated with this array
    Raises:
        TypeError: if cannot handle  image type
    References:
        https://github.com/python-pillow/Pillow/blob/master/src/PIL/Image.py#L2788
    """
    try:
        typekey = (1, 1) + arr.shape[2:], arr.dtype.str
    except KeyError as e:
        raise TypeError("Cannot handle this data type: %s" % arr.dtype.str) from e
    try:
        imgType, _ = imageTypeMap[typekey]
        return imgType
    except KeyError as e:
        raise TypeError("Cannot handle this data type: %s, %s" % typekey) from e


def pilToNumpy(img: Image) -> np.ndarray:
    """
    Fast load pillow image to numpy array
    Args:
        img: pillow image
    Returns:
        numpy array which alignment by 32 bit
    Raises:
        RuntimeError: if encoding failed
    References:
        https://habr.com/ru/post/545850/
    """
    img.load()
    # unpack data
    e = PIL.Image._getencoder(img.mode, "raw", img.mode)
    e.setimage(img.im)

    # NumPy buffer for the result
    shape, typestr = PIL.Image._conv_type_shape(img)
    size = shape[0] * shape[1] * shape[2]
    # faceengine require alignment by 32 bits
    data = np.empty(size + 32, dtype=np.dtype(typestr, align=True))
    cdata = data.ctypes.data
    if cdata % 32 != 0:
        offsetForAlignment = 32 - cdata % 32
    else:
        offsetForAlignment = 0
    dData = data.data
    mem = dData.cast("B", (dData.nbytes,))

    bufsize, s, offset = 65536, 0, offsetForAlignment
    encode = e.encode
    while not s:
        l, s, d = encode(bufsize)
        mem[offset : offset + l] = d  # noqa: E203
        offset += l
    data = data[offsetForAlignment : offsetForAlignment + size]  # noqa: E203
    data = data.reshape(shape)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data
