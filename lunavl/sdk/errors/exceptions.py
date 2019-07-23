"""
Module realizes LunaSDKException - single exception for rising in sdk module
"""
from functools import wraps
from typing import Optional, Any

from lunavl.sdk.errors.errors import ErrorInfo


class LunaSDKException(Exception):
    """
    SDK Exception

    Attributes:
        error (ErrorInfo): error
        context (Any): additional info
    """

    def __init__(self, error: ErrorInfo, context: Optional[Any] = None, exception: Optional[Exception] = None):
        super().__init__(str(error))
        self.exception = exception
        self.error = error
        self.context = context


def CoreExceptionWarp(error: ErrorInfo):
    """
    Decorator for catching exceptions  from c.

    Args:
        error: returning error in the exception case
    Returns:
        if exception was caught, system calls error method with error
    """

    def realWarp(func):
        @wraps(func)
        async def wrap(*func_args, **func_kwargs):
            try:
                res = await func(*func_args, **func_kwargs)
                return res
            except RuntimeError as e:
                raise LunaSDKException(error.detalize(str(e)), exception=e)
        return wrap
    return realWarp
