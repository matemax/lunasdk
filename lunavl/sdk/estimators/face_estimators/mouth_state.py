"""Module contains a mouth state estimator

see `mouth state`_
"""
from typing import Union, Dict

from FaceEngine import ISmileEstimatorPtr, SmileEstimation  # pylint: disable=E0611,E0401
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import CoreExceptionWrap, LunaSDKException

from lunavl.sdk.base import BaseEstimation
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
    def __init__(self, coreEstimation: SmileEstimation):
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
    def mouth(self) -> float:
        """
        Get mouth score value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.mouth

    @property
    def occlusion(self) -> float:
        """
        Get occlusion score value.

        Returns:
            value in range [0, 1]
        """
        return self._coreEstimation.occlusion

    def asDict(self) -> Dict[str, float]:
        """
        Convert to dict.

        Returns:
            {'score': self.mouth, 'occlusion': self.occlusion, 'smile': self.smile}
        """
        return {"score": self.mouth, "occluded": self.occlusion, "smile": self.smile}


class MouthStateEstimator(BaseEstimator):
    """
    Mouth state estimator.
    """

    #  pylint: disable=W0235
    def __init__(self, coreEstimator: ISmileEstimatorPtr):
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
        if error.isError:
            raise LunaSDKException(LunaVLError.fromSDKError(error))
        return MouthStates(mouthState)
