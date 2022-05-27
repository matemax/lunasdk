from enum import Enum
from typing import Optional

import FaceEngine as CoreFE  # pylint: disable=E0611,E0401


class DeviceClass(Enum):
    """
    Device enum
    """

    cpu = "cpu"
    gpu = "gpu"


class LaunchOptions:
    """
    Estimator launch options. Some parameters are set for future evaluations.

    Parameters:
      deviceClass: type of device for estimation performing.
      deviceId: device number, actual for gpu and npu
      runConcurrently:

    Attributes:
        _coreLaunchOptions: core launch options
    """

    def __init__(
        self,
        deviceClass: Optional[DeviceClass] = None,
        deviceId: Optional[int] = None,
        runConcurrently: Optional[bool] = None,
    ):
        self._coreLaunchOptions = CoreFE.LaunchOptions()
        if deviceClass:
            if deviceClass == DeviceClass.gpu:
                device = CoreFE.DeviceClass.GPU
            else:
                device = CoreFE.DeviceClass.CPU_AVX2
        else:
            device = CoreFE.DeviceClass.CPU_AVX2
        self._coreLaunchOptions.deviceClass = device
        if deviceId:
            self._coreLaunchOptions.deviceClass = deviceId
        if runConcurrently is not None:
            self._coreLaunchOptions.runConcurrently = runConcurrently
        else:
            self._coreLaunchOptions.runConcurrently = True

    @property
    def deviceClass(self) -> DeviceClass:
        """Get device class"""
        return DeviceClass(self._coreLaunchOptions.deviceClass.name.lower())

    @property
    def coreLaunchOptions(self) -> CoreFE.LaunchOptions:
        """Get core launch options"""
        return self._coreLaunchOptions
