from lunavl.sdk.faceengine.engine import VLFaceEngine
from lunavl.sdk.faceengine.setting_provider import RuntimeSettingsProvider
from lunavl.sdk.launch_options import DeviceClass, LaunchOptions
from tests.base import BaseTestClass


class TestLaunchOptions(BaseTestClass):
    """
    Test launch options
    """

    def test_get_launch_options(self):
        """Get launch options from VLFaceEngine"""
        runtime = RuntimeSettingsProvider()
        for device in DeviceClass:
            with self.subTest(device):
                runtime.runtimeSettings.deviceClass = device
                fe = VLFaceEngine(runtimeConf=runtime)
                assert fe.getLaunchOptions(None).deviceClass == device

    def test_default_initialization_launch_options(self):
        """Default launch options initialization"""
        lo = LaunchOptions()
        assert DeviceClass.cpu == lo.deviceClass
        assert lo.runConcurrently
        assert -1 == lo.deviceId

    def test_initialization_launch_options(self):
        """Test launch options initialization"""
        for device in DeviceClass:
            for runConcurrently in (True, False):
                for deviceId in (-2, -1, 1):
                    with self.subTest(device=device, runConcurrently=runConcurrently, deviceId=deviceId):
                        lo = LaunchOptions(deviceClass=device, runConcurrently=runConcurrently, deviceId=deviceId)
                        assert deviceId == lo.deviceId
                        assert runConcurrently == lo.runConcurrently
                        assert device == lo.deviceClass

    def test_launch_options_device_0(self):
        """Initialize launch options with deviceId=0"""
        lo = LaunchOptions(deviceId=0)
        assert -1 == lo.deviceId
