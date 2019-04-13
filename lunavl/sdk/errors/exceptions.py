from typing import Optional, Any

from lunavl.sdk.errors.errors import Error


class LunaSDKException(Exception):
    def __init__(self, error: Error, context: Optional[Any] = None):
        super().__init__(str(error))
        self.error = error
        self.context = context
