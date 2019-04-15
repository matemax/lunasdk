"""
Module contains class ErrorInfo. Structure for errors.
"""
from FaceEngine import FSDKErrorResult


class ErrorInfo:
    """
    Error info

    Attributes:
        errorCode (int): error code
        desc (str): error description
        detail (str): detail
    """
    __slots__ = ["errorCode", "desc", "detail"]

    def __init__(self, errorCode: int, desc: str, detail: str):
        """
        Init

        Args:
            errorCode: error code
            desc: description
            detail: detail
        """
        self.errorCode = errorCode
        self.desc = desc
        self.detail = detail

    @classmethod
    def fromSDKError(cls, errorCode: int, desc: str, sdkError: FSDKErrorResult):
        error = cls(errorCode, desc, sdkError.what)
        return error

    def asDict(self):
        return {"error_code": self.errorCode, "desc": self.desc, "detail": self.detail}

    def __repr__(self) -> str:
        """
        Error representation.

        Returns:
            "error code: {self.errorCode}, desc: {self.desc}, detail {self.detail}"

        >>> ErrorInfo(123, "Test", "Test error")
        error code: 123, desc: Test, detail Test error
        """
        return "error code: {}, desc: {}, detail {}".format(self.errorCode, self.desc, self.detail)
