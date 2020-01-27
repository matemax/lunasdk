"""Module for creating warped images
"""
from typing import Union, Optional

from FaceEngine import IWarperPtr, Transformation  # pylint: disable=E0611,E0401
from FaceEngine import Image as CoreImage  # pylint: disable=E0611,E0401
from PIL.Image import Image as PilImage
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException, CoreExceptionWrap
from lunavl.sdk.faceengine.facedetector import FaceDetection, Landmarks68, Landmarks5
from lunavl.sdk.image_utils.image import VLImage, ColorFormat


class WarpedImage(VLImage):
    """
    Raw warped image.

    Properties of a warped image:

        - its size is always 250x250 pixels
        - it's always in RGB color format
        - it always contains just a single face
        - the face is always centered and rotated so that imaginary line between the eyes is horizontal.
    """

    def __init__(
        self,
        body: Union[bytes, bytearray, PilImage, CoreImage, VLImage],
        filename: str = "",
        colorFormat: Optional[ColorFormat] = None,
    ):
        """
        Init.

        Args:
            body: body of image - bytes/bytearray or core image or pil image or vlImage
            filename: user mark a source of image
            colorFormat: output image color format
        """
        if isinstance(body, VLImage):
            self.source = body.source
            self.filename = body.filename
            self.coreImage = body.coreImage
        else:
            super().__init__(body, filename=filename, colorFormat=colorFormat)
        self.assertWarp()

    def assertWarp(self):
        """
        Validate size and format

        Raises:
            ValueError("Bad image size for warped image"): if image has incorrect size
            ValueError("Bad image format for warped image, must be R8G8B8"): if image has incorrect format
        Warnings:
            this checks are not guarantee that image is warp. This function is intended for debug
        """
        if self.rect.size.height != 250 or self.rect.width != 250:
            raise ValueError("Bad image size for warped image")
        if self.format != self.format.R8G8B8:
            raise ValueError("Bad image format for warped image, must be R8G8B8")

    #  pylint: disable=W0221
    @classmethod
    def load(cls, *_, filename: Optional[str] = None, url: Optional[str] = None) -> "WarpedImage":  # type: ignore
        """
        Load imag from numpy array or file or url.

        Args:
            *_: for remove positional argument
            filename: filename
            url: url

        Returns:
            warp
        """
        warp = cls(body=VLImage.load(filename=filename, url=url), filename=filename or "")
        warp.assertWarp()
        return warp

    @property
    def warpedImage(self) -> "WarpedImage":
        """
        Property for compatibility with *Warp* for outside methods.
        Returns:
            self
        """
        return self


class Warp:
    """
    Structure for storing warp.

    Attributes:
        sourceDetection (FaceDetection): detection which generated warp
        warpedImage (WarpedImage):
    """

    __slots__ = ["sourceDetection", "warpedImage"]

    def __init__(self, warpedImage: WarpedImage, sourceDetection: FaceDetection):
        """
        Init.

        Args:
            warpedImage: warped image
            sourceDetection: detection which generated warp
        """
        self.sourceDetection = sourceDetection
        self.warpedImage = warpedImage


class Warper:
    """
    Class warper.

    Attributes:
        _coreWarper (IWarperPtr): core warper
    """

    __slots__ = ["_coreWarper"]

    def __init__(self, coreWarper: IWarperPtr):
        """
        Init.

        Args:
            coreWarper: core warper
        """
        self._coreWarper = coreWarper

    @CoreExceptionWrap(LunaVLError.WarpTransformationError)
    def _createWarpTransformation(self, faceDetection: FaceDetection) -> Transformation:
        """
        Create warp transformation.

        Args:
            faceDetection: face detection with landmarks5

        Returns:
            transformation
        Raises:
            ValueError: if detection does not contain a landmarks5
        """
        if faceDetection.landmarks5 is None:
            raise ValueError("detection must contains landmarks5")
        return self._coreWarper.createTransformation(
            faceDetection.coreEstimation.detection, faceDetection.landmarks5.coreEstimation
        )

    @CoreExceptionWrap(LunaVLError.CreationWarpError)
    def warp(self, faceDetection: FaceDetection) -> Warp:
        """
        Create warp from detection.

        Args:
            faceDetection: face detection with landmarks5

        Returns:
            Warp
        Raises:
            LunaSDKException: if creation failed
        """
        transformation = self._createWarpTransformation(faceDetection)
        error, warp = self._coreWarper.warp(faceDetection.image.coreImage, transformation)
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))

        warpedImage = WarpedImage(body=warp, filename=faceDetection.image.filename)

        return Warp(warpedImage, faceDetection)

    def makeWarpTransformationWithLandmarks(
        self, faceDetection: FaceDetection, typeLandmarks: str
    ) -> Union[Landmarks68, Landmarks5]:
        """
        Make warp transformation with landmarks

        Args:
            faceDetection: face detection  with landmarks5
            typeLandmarks: landmarks for warping ("L68" or "L5")

        Returns:
            warping landmarks
        Raises:
            ValueError: if landmarks5 is not estimated
            LunaSDKException: if transform failed
        """
        transformation = self._createWarpTransformation(faceDetection)
        if typeLandmarks == "L68":
            if faceDetection.landmarks68 is None:
                raise ValueError("landmarks68 does not estimated")
            error, warp = self._coreWarper.warp(faceDetection.landmarks68.coreEstimation, transformation)
        elif typeLandmarks == "L5":
            if faceDetection.landmarks5 is None:
                raise ValueError("landmarks5 does not estimated")
            error, warp = self._coreWarper.warp(faceDetection.landmarks5.coreEstimation, transformation)
        else:
            raise ValueError("Invalid value of typeLandmarks, must be 'L68' or 'L5'")
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        if typeLandmarks == "L68":
            return Landmarks68(warp)
        return Landmarks5(warp)
