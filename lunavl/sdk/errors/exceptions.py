"""
Module realizes LunaSDKException - single exception for rising in sdk module
"""
from typing import Optional, Any

from lunavl.sdk.errors.errors import ErrorInfo


class LunaSDKException(Exception):
    """
    SDK Exception

    Attributes:
        error (ErrorInfo): error
        context (Any): additional info
    """
    def __init__(self, error: ErrorInfo, context: Optional[Any] = None):
        super().__init__(str(error))
        self.error = error
        self.context = context
