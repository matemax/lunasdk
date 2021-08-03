import PIL.ImageFile


def applyLoadTruncateImages():
    """
    Allow load truncated images
    """
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
