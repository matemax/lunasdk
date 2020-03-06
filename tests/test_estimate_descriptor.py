"""
Test estimate descriptor.
"""
import unittest

from sdk.estimators.face_estimators.warper import WarpedImage
from sdk.faceengine.descriptors import FaceDescriptor
from tests.base import BaseTestClass
from tests.resources import WARP_WHITE_MAN

EDVa = EXISTENT_DESCRIPTOR_VERSION_ABUNDANCE = [46, 52, 54, 56]

warp = WarpedImage.load(filename=WARP_WHITE_MAN)


class TestEstimateDescriptor(BaseTestClass):
    """
    Test extract descriptor.
    """

    @staticmethod
    def assert_descriptor(expectedVersion: int, descriptor: FaceDescriptor) -> None:
        """
        Assert extracted descriptor.

        Args:
            expectedVersion: expected descriptor version
            descriptor: extracted descriptor
        """
        assert descriptor.model == expectedVersion, "descriptor has wrong version"
        length = {46: 256, 52: 256, 54: 512, 56: 512}[expectedVersion]
        assert len(descriptor.asBytes) == length
        assert len(descriptor.asVector) == length
        assert len(descriptor.rawDescriptor) == length + 8

    def test_create_estimators(self):
        """
        Test create estimators of different descriptor versions.
        """
        for planVersion in range(min(EDVa) - 10, max(EDVa) + 10):
            with self.subTest(descriptor_version=planVersion):
                try:
                    self.faceEngine.createFaceDescriptorEstimator(descriptorVersion=planVersion)
                except RuntimeError as e:
                    if planVersion in EDVa:
                        raise AssertionError(f"Descriptor version {planVersion} is not supported. But must be.") from e
                else:
                    if planVersion not in EDVa:
                        raise AssertionError(f"Descriptor version {planVersion} is supported. But should not.")

    def test_extract_descriptors_positive(self):
        """
        Test correctly extract descriptor.
        """
        for planVersion in EDVa:
            for kw in (
                dict(),
                dict(descriptor=self.faceEngine.createFaceDescriptorFactory(planVersion).generateDescriptor()),
            ):
                extractor = self.faceEngine.createFaceDescriptorEstimator(descriptorVersion=planVersion)
                with self.subTest(planVersion=planVersion, external_descriptor=bool(kw)):
                    descriptor = extractor.estimate(warp, **kw)
                    self.assert_descriptor(planVersion, descriptor)

    @unittest.skip("dont do it FSDK-2186")
    def test_extract_descriptors_incorrect_source_descriptors(self):
        """
        Test extract descriptor using incorrect source descriptor.
        """
        for planVersion in EDVa:
            extractor = self.faceEngine.createFaceDescriptorEstimator(descriptorVersion=planVersion)
            for descriptorVersion in set(EDVa) - {planVersion}:
                descriptorOfAnotherVersion = self.faceEngine.createFaceDescriptorFactory(
                    descriptorVersion=descriptorVersion
                ).generateDescriptor()
                with self.subTest(planVersion=planVersion, descriptorVersion=descriptorVersion):
                    print(f"Plan {planVersion}, empty descriptor version {descriptorVersion}")
                    descriptor = extractor.estimate(warp, descriptor=descriptorOfAnotherVersion)
                    self.assert_descriptor(planVersion, descriptor)
