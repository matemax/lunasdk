"""
Module contains a mouth state estimator
"""
from typing import Union

from FaceEngine import ISmileEstimatorPtr, SmileEstimation  # pylint: disable=E0611,E0401

from lunavl.sdk.faceengine.warper import Warp, WarpedImage


class MouthStates:
    """
    Mouth states. There are 3 states of mouth: smile, occlusion and neither a smile nor an occlusion was detected.

    Attributes:
        _coreEstimation (SmileEstimation): core mouth estimation.
    """
    __slots__ = ['_coreEstimation']

    def __init__(self, coreEstimation: SmileEstimation):
        self._coreEstimation = coreEstimation

    @property
    def smile(self) -> float:
        """
        Get smile score value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.smile

    @property
    def mouth(self):
        """
        Get mouth score value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.mouth

    @property
    def occlusion(self):
        """
        Get occlusion score value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.occlusion

    def asDict(self):
        """
        Convert ot dict.

        Returns:
            {'score': self.mouth, 'occlusion': self.occlusion, 'smile': self.smile}
        """
        return {'score': self.mouth, 'occlusion': self.occlusion, 'smile': self.smile}

    def __repr__(self):
        """
        Representation.

        Returns:
            str(self.asDict())
        """
        return str(self.asDict())


class MouthStateEstimator:
    """
    Mouth state estimator.

    Attributes:
        _coreEstimator (IEmotionsEstimatorPtr): core estimator
    """
    __slots__ = ['_coreEstimator']

    def __init__(self, coreEstimator: ISmileEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        self._coreEstimator = coreEstimator

    def estimate(self, warp: Union[Warp, WarpedImage]) -> MouthStates:
        """
        Estimate mouth state on warp.

        Args:
            warp: warped image

        Returns:
            estimated states
        """
        error, mouthState = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        if error.isError:
            raise ValueError("12343")
        return MouthStates(mouthState)
