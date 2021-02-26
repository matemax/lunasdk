"""
Module contains class ErrorInfo. Structure for errors.
"""
import inspect
from typing import Dict, Union
from FaceEngine import FSDKErrorResult  # pylint: disable=E0611,E0401


class ErrorInfo:
    """
    Error info

    Attributes:
        errorCode (int): error code
        desc (str): error description
        detail (str): detail
    """

    __slots__ = ["errorCode", "description", "detail"]

    def __init__(self, errorCode: int, desc: str, detail: str):
        """
        Init

        Args:
            errorCode: error code
            desc: description
            detail: detail
        """
        self.errorCode = errorCode
        self.description = desc
        self.detail = detail

    def asDict(self) -> Dict[str, Union[int, str]]:
        """
        Convert  to dict.

        Returns:
            {"error_code": self.errorCode, "desc": self.desc, "detail": self.detail}

        >>> ErrorInfo(123, "Test", "Test error").asDict()
        {'error_code': 123, 'desc': 'Test', 'detail': 'Test error'}
        """
        return {"error_code": self.errorCode, "desc": self.description, "detail": self.detail}

    def __repr__(self) -> str:
        """
        Error representation.

        Returns:
            "error code: {self.errorCode}, desc: {self.desc}, detail {self.detail}"

        >>> ErrorInfo(123, "Test", "Test error")
        error code: 123, desc: Test, detail: Test error
        """
        return "error code: {}, desc: {}, detail: {}".format(self.errorCode, self.description, self.detail)

    def format(self, details: str) -> "ErrorInfo":
        return ErrorInfo(self.errorCode, self.description, details)


class LunaVLError:
    UnknownError = ErrorInfo(99999, "Unknown fsdk core error", "")

    Ok = ErrorInfo(100000, "Ok", "")
    BufferIsEmpty = ErrorInfo(100001, "Buffer is empty", "")
    BufferIsNull = ErrorInfo(100002, "Buffer is null", "")
    BufferIsFull = ErrorInfo(100003, "Buffer is full", "")
    IncompatibleDescriptors = ErrorInfo(100004, "Descriptors are incompatible", "")
    Internal = ErrorInfo(100005, "Internal error", "")
    InvalidBufferSize = ErrorInfo(100006, "Invalid buffer size", "")
    InvalidDescriptor = ErrorInfo(100007, "Invalid descriptor", "")
    InvalidDescriptorBatch = ErrorInfo(100008, "Invalid descriptor batch", "")
    InvalidDetection = ErrorInfo(100009, "Invalid detection", "")
    InvalidImage = ErrorInfo(100010, "Invalid image", "")
    InvalidImageFormat = ErrorInfo(100011, "Invalid image format", "")
    InvalidImageSize = ErrorInfo(100012, "Invalid image size", "")

    InvalidInput = ErrorInfo(100013, "Invalid input", "")
    InvalidLandmarks5 = ErrorInfo(100014, "Invalid landmarks 5", "")
    InvalidLandmarks68 = ErrorInfo(100015, "Invalid landmarks 68", "")
    InvalidRect = ErrorInfo(100016, "Invalid rectangle", "")
    InvalidSettingsProvider = ErrorInfo(100017, "Invalid settings provider", "")
    LicenseError = ErrorInfo(100018, "Licensing issue", "")
    ModuleNotInitialized = ErrorInfo(100019, "Module is not initialized", "")
    ModuleNotReady = ErrorInfo(100020, "Module is not ready", "")

    FailedToInitialize = ErrorInfo(100021, "Error during initialization fdsk image", "")
    FailedToLoad = ErrorInfo(100022, "Error during image loadin", "")
    FailedToSave = ErrorInfo(100023, "Error during image saving", "")
    InvalidArchive = ErrorInfo(100024, "Archive image error", "")
    InvalidBitmap = ErrorInfo(100025, "Invalid detection", "")
    InvalidConversion = ErrorInfo(100026, "Image conversion not implemented", "")
    InvalidDataPtr = ErrorInfo(100027, "Bad input image data pointer.", "")
    InvalidDataSize = ErrorInfo(100028, "Bad input image data size", "")

    InvalidFormat = ErrorInfo(100029, "Unsupported image format", "")
    InvalidHeight = ErrorInfo(100030, "Invalid image height", "")
    InvalidPath = ErrorInfo(100031, "Bad path for image saving / loading", "")
    InvalidMemory = ErrorInfo(100032, "Error at image memory opening", "")
    InvalidType = ErrorInfo(100033, "Unsupported image type", "")
    InvalidWidth = ErrorInfo(100034, "Invalid image width", "")
    BatchedInternalError = ErrorInfo(100035, "Batching error", "")

    CreationDescriptorError = ErrorInfo(110001, "Creation descriptor error", "")
    CreationBatchDescriptorsError = ErrorInfo(110002, "Creation descriptor error", "")
    CreationImageError = ErrorInfo(110003, "Creation core image error", "")
    EstimationDescriptorError = ErrorInfo(110004, "Estimation descriptor error", "")
    EstimationBatchDescriptorError = ErrorInfo(110005, "Estimation descriptor error", "")
    EstimationBasicAttributeError = ErrorInfo(110006, "Estimation basic attributes error", "")
    BatchEstimationBasicAttributeError = ErrorInfo(110007, "Batch estimation basic attributes error", "")
    EstimationAGSError = ErrorInfo(110008, "Estimation AGS error", "")
    EstimationHeadPoseError = ErrorInfo(110009, "Estimation head pose error", "")
    EstimationEthnisitiesError = ErrorInfo(1100010, "Estimation ethnities error", "")
    EstimationEyesGazeError = ErrorInfo(110011, "Estimation eyes gase error", "")
    EstimationEmotionsError = ErrorInfo(110012, "Estimation emotions error", "")
    EstimationWarpQualityError = ErrorInfo(110013, "Estimation warp quality error", "")
    EstimationMouthStateError = ErrorInfo(110014, "Estimation mouth state error", "")
    EstimationEyesError = ErrorInfo(110015, "Estimation eyes error", "")
    CreationWarpError = ErrorInfo(110016, "Creation warped image error", "")
    WarpTransformationError = ErrorInfo(110017, "Landmarks transformation error", "")
    DetectOneFaceError = ErrorInfo(110018, "Detect one face error", "")
    DetectFacesError = ErrorInfo(110019, "Detect faces error", "")
    HighMemoryUsage = ErrorInfo(110020, "High memory usage", "")
    DetectHumanError = ErrorInfo(110021, "Detect one human body error", "")
    DetectHumansError = ErrorInfo(110022, "Detect humans bodies error", "")
    EstimationMaskError = ErrorInfo(110023, "Estimation mask error", "")
    BadAggregationThreshold = ErrorInfo(110024, "Filtered aggregation error", "")
    EstimationGlassesError = ErrorInfo(110025, "Estimation glasses error", "")
    CredibilityCheckError = ErrorInfo(110026, "Estimation credibility check error", "")

    @classmethod
    def fromSDKError(cls, sdkError: FSDKErrorResult) -> "ErrorInfo":
        """
        Create error from sdk error

        Args:
            sdkError: sdk error

        Returns:
            error, detail is what of sdk error
        """

        errorClassErrors = inspect.getmembers(cls, lambda err: isinstance(err, ErrorInfo))

        error = sdkError.error
        for errorName, errorVal in errorClassErrors:
            if errorName == error.name:
                return ErrorInfo(errorVal.errorCode, errorVal.description, sdkError.what)

        return ErrorInfo(cls.UnknownError.errorCode, cls.UnknownError.description, sdkError.what)
