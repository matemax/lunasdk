from FaceEngine import FSDKErrorResult


class Error:
    __slots__ = ["errorCode", "desc", "detail"]

    def __init__(self, errorCode: int, desc: str, detail: str):
        self.errorCode = errorCode
        self.desc = desc
        self.detail = detail

    @classmethod
    def fromSDKError(cls, errorCode: int, desc: str, sdkError: FSDKErrorResult):
        error = cls(errorCode, desc, sdkError.what)
        return error

    def asDict(self):
        return {"error_code": self.errorCode, "desc": self.desc, "detail": self.detail}

    def __repr__(self):
        return "error code: {}, desc: {}, detail {}".format(self.errorCode, self.desc, self.detail)
