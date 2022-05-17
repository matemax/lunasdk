"""
Module realizes LunaSDKException - single exception for rising in sdk module
"""
from typing import Any, List, Optional

from FaceEngine import FSDKErrorResult  # pylint: disable=E0611,E0401

from lunavl.sdk.errors.errors import ErrorInfo, LunaVLError


class LunaSDKException(Exception):
    """
    SDK Exception

    Attributes:
        error (ErrorInfo): error
        context (Any): additional info
        exception (Optional[Exception]): other exception which converted to self
    """

    def __init__(self, error: ErrorInfo, context: Optional[Any] = None, exception: Optional[Exception] = None):
        super().__init__(str(error))
        self.exception = exception
        self.error = error
        self.context = context


def assertError(error: FSDKErrorResult, context: Optional[List[Any]] = None) -> None:
    """
    Assert core optional.
    Args:
        error: optional
        context: list with errors

    Raises:
        LunaSDKException: if optional contains error

    """
    if error.isError:
        raise LunaSDKException(LunaVLError.fromSDKError(error), context)
