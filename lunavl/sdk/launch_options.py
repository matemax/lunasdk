from enum import Enum
from typing import Optional

import FaceEngine as CoreFE  # pylint: disable=E0611,E0401


class DeviceClass(Enum):
    """
    Device enum
    """

    cpu = "cpu"
    gpu = "gpu"
    # npu = "npu"


class LaunchOptions:
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
            # elif deviceClass == DeviceClass.npu:
            #     device = CoreFE.DeviceClass.NPU
            else:
                device = CoreFE.DeviceClass.CPU
            self._coreLaunchOptions.deviceClass = device
        if deviceId:
            self._coreLaunchOptions.deviceClass = deviceId
        if runConcurrently is not None:
            self._coreLaunchOptions.runConcurrently = runConcurrently

    @property
    def deviceClass(self):
        return DeviceClass(self._coreLaunchOptions.deviceClass.name.lower())

    @property
    def coreLaunchOptions(self):
        return self._coreLaunchOptions
