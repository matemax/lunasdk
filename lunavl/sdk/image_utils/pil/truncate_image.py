"""
Module allows enable pillow  to work with truncated images

References:
    https://pillow.readthedocs.io/en/stable/reference/ImageFile.html?#PIL.ImageFile.LOAD_TRUNCATED_IMAGES
"""
import PIL.ImageFile


def applyLoadTruncateImages():
    """
    Allow load truncated images
    """
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
