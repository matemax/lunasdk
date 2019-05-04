from typing import Union


class BaseEstimation:
    """
    Base class for estimation structures.

    Attributes:
        _coreEstimation: core estimation
    """
    __slots__ = ("_coreEstimation",)

    def __init__(self, coreEstimation):
        self._coreEstimation = coreEstimation

    @property
    def coreEstimation(self):
        """
        Get core estimation from init
        Returns:
            _coreEstimation
        """
        return self._coreEstimation

    def asDict(self) -> Union[dict, list]:
        """
        Convert to  dict.

        Returns:
            dict from luna api
        """
        raise NotImplemented

    def __repr__(self) -> str:
        """
        Representation.

        Returns:
            str(self.asDict())
        """
        return str(self.asDict())
