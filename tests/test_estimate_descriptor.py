"""
Test estimate descriptor.
"""
import unittest

from lunavl.sdk.estimators.face_estimators.warper import WarpedImage
from lunavl.sdk.faceengine.descriptors import FaceDescriptor
from tests.base import BaseTestClass
from tests.resources import WARP_WHITE_MAN

EDVa = EXISTENT_DESCRIPTOR_VERSION_ABUNDANCE = [46, 52, 54, 56]

warp = WarpedImage.load(filename=WARP_WHITE_MAN)
warps = [warp] * 3


class TestEstimateDescriptor(BaseTestClass):
    """
    Test estimate descriptor.
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

    def test_create_estimators_positive(self):
        """
        Test create estimators of different existent plan versions.
        """
        for planVersion in EDVa:
            with self.subTest(descriptor_version=planVersion):
                try:
                    self.faceEngine.createFaceDescriptorEstimator(descriptorVersion=planVersion)
                except RuntimeError as e:
                    raise AssertionError(f"Descriptor version {planVersion} is not supported. But must be.") from e

    def test_extract_descriptors_positive(self):
        """
        Test correctly estimate descriptor.
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

    def test_create_estimators_negative(self):
        """
        Test create estimators of different nonexistent plan versions.
        """
        nonexistentVersions = set(range(min(EDVa) - 10, max(EDVa) + 10)) - set(EDVa)
        for planVersion in nonexistentVersions:
            with self.subTest(descriptor_version=planVersion):
                try:
                    self.faceEngine.createFaceDescriptorEstimator(descriptorVersion=planVersion)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(f"Descriptor version {planVersion} is supported. But should not.")

    @unittest.skip("dont do it FSDK-2186")
    def test_extract_descriptors_incorrect_source_descriptors(self):
        """
        Test estimate descriptor using incorrect source descriptor.
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

    def test_extract_descriptors_batch_positive(self):
        """
        Test correctly estimate descriptor batch.
        """
        for planVersion in EDVa:
            extractor = self.faceEngine.createFaceDescriptorEstimator(descriptorVersion=planVersion)
            for kw in (
                dict(),
                dict(
                    descriptorBatch=self.faceEngine.createFaceDescriptorFactory(planVersion).generateDescriptorsBatch(
                        len(warps)
                    )
                ),
            ):
                for aggregate in (True, False):
                    with self.subTest(planVersion=planVersion, aggregate=aggregate, external_descriptor=bool(kw)):
                        descriptorsRaw, descriptorAggregated = extractor.estimateDescriptorsBatch(
                            warps, aggregate=aggregate, **kw
                        )
                        for descriptorRaw in descriptorsRaw:
                            self.assert_descriptor(planVersion, descriptorRaw)

                        if aggregate:
                            self.assert_descriptor(planVersion, descriptorAggregated)
                        else:
                            self.assertIsNone(descriptorAggregated)

    @unittest.skip("dont do it FSDK-2186")
    def test_extract_descriptors_batch_incorrect_source_descriptors(self):
        """
        Test correctly estimate descriptor batch.
        """
        for planVersion in EDVa:
            extractor = self.faceEngine.createFaceDescriptorEstimator(descriptorVersion=planVersion)
            for descriptorVersion in set(EDVa) - {planVersion}:
                for kw in (
                    dict(),
                    dict(
                        descriptorBatch=self.faceEngine.createFaceDescriptorFactory(
                            descriptorVersion
                        ).generateDescriptorsBatch(len(warps))
                    ),
                ):
                    for aggregate in (True, False):
                        with self.subTest(planVersion=planVersion, aggregate=aggregate, external_descriptor=bool(kw)):
                            descriptorsRaw, descriptorAggregated = extractor.estimateDescriptorsBatch(
                                warps, aggregate=aggregate, **kw
                            )
                            for descriptorRaw in descriptorsRaw:
                                self.assert_descriptor(planVersion, descriptorRaw)

                            if aggregate:
                                self.assert_descriptor(planVersion, descriptorAggregated)
                            else:
                                self.assertIsNone(descriptorAggregated)
