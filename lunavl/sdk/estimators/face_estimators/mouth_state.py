"""Module contains a mouth state estimator

see `mouth state`_
"""
from typing import Union, Dict

from FaceEngine import MouthEstimation, IMouthEstimatorPtr  # pylint: disable=E0611,E0401

from lunavl.sdk.base import BaseEstimation
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, assertError
from ..base import BaseEstimator
from ..face_estimators.facewarper import FaceWarp, FaceWarpedImage


class MouthStates(BaseEstimation):
    """
    Mouth states. There are 3 states of mouth: smile, occlusion and neither a smile nor an occlusion was detected.

    Estimation properties:

        - smile
        - mouth
        - occlusion
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimation: MouthEstimation):
        super().__init__(coreEstimation)

    @property
    def smile(self) -> float:
        """
        Get smile score value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.smile

    @property
    def opened(self) -> float:
        """
        Get opened score value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.opened

    @property
    def occlusion(self) -> float:
        """
        Get occlusion score value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.occluded

    def asDict(self) -> Dict[str, float]:
        """
        Convert to dict.

        Returns:
            {'opened': self.opened, 'occlusion': self.occlusion, 'smile': self.smile}
        """
        return {"opened": self.opened, "occluded": self.occlusion, "smile": self.smile}


class MouthStateEstimator(BaseEstimator):
    """
    Mouth state estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: IMouthEstimatorPtr):
        """
        Init.

        Args:
            coreEstimator: core estimator
        """
        super().__init__(coreEstimator)

    #  pylint: disable=W0221
    @CoreExceptionWrap(LunaVLError.EstimationMouthStateError)
    def estimate(self, warp: Union[FaceWarp, FaceWarpedImage]) -> MouthStates:
        """
        Estimate mouth state on warp.

        Args:
            warp: warped image

        Returns:
            estimated states
        Raises:
            LunaSDKException: if estimation failed
        """
        error, mouthState = self._coreEstimator.estimate(warp.warpedImage.coreImage)
        assertError(error)
        return MouthStates(mouthState)
