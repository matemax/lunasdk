"""
Module realizes LunaSDKException - single exception for rising in sdk module
"""
from functools import wraps
from typing import Optional, Any, Callable

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


def CoreExceptionWrap(error: ErrorInfo):
    """
    Decorator catch runtime exceptions from core (as supposed)  and converts it to LunaSDKException.

    Args:
        error: returning error in the exception case
    Returns:
        if exception was caught, system calls error method with error
    """

    def realWarp(func: Callable):
        @wraps(func)
        def wrap(*func_args, **func_kwargs):
            try:
                res = func(*func_args, **func_kwargs)
                return res
            except RuntimeError as e:
                raise LunaSDKException(error.format(str(e)), exception=e)

        return wrap

    return realWarp


def assertError(error: FSDKErrorResult) -> None:
    """
    Assert core optional.
    Args:
        error: optional

    Raises:
        LunaSDKException: if optional contains error

    """
    if error.isError:
        raise LunaSDKException(LunaVLError.fromSDKError(error))
