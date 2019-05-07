"""Module with base classes of estimators and estimations
"""
from abc import ABC, abstractmethod
from typing import Union, Any


class BaseEstimation(ABC):
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

    @abstractmethod
    def asDict(self) -> Union[dict, list]:
        """
        Convert to  dict.

        Returns:
            dict from luna api
        """
        pass

    def __repr__(self) -> str:
        """
        Representation.

        Returns:
            str(self.asDict())
        """
        return str(self.asDict())


class BaseEstimator(ABC):
    """
    Base estimator class. Class is  a container for core estimations. Mostly estimate attributes  can be get through
    a corresponding properties.

    Attributes:
        _coreEstimator: core estimator
    """
    __slots__ = ('_coreEstimator',)

    def __init__(self, coreEstimator):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        self._coreEstimator = coreEstimator

    @abstractmethod
    def estimate(self, *args, **kwargs) -> Any:
        """
        Estimate attributes on warp.

        Returns:
            estimated attributes
        """
        pass
