"""
Test estimate descriptor.
"""
import unittest

import pytest

from lunavl.sdk.estimators.face_estimators.warper import WarpedImage
from lunavl.sdk.faceengine.descriptors import FaceDescriptor, FaceDescriptorBatch
from tests.base import BaseTestClass
from tests.resources import WARP_WHITE_MAN

EDVa = EXISTENT_DESCRIPTOR_VERSION_ABUNDANCE = [46, 52, 54, 56]

warp = WarpedImage.load(filename=WARP_WHITE_MAN)
warps = [warp] * 3


class TestDescriptorFunctionality(BaseTestClass):
    version = EDVa[0]
    _estimator = BaseTestClass.faceEngine.createFaceDescriptorEstimator(version)
    descriptor = _estimator.estimate(warp)
    descriptorBatch, aggregatedDescriptor = _estimator.estimateDescriptorsBatch(warps, aggregate=True)

    def assertDescriptor(self, descriptor: FaceDescriptor):
        """
        Assert descriptor format.

        Args:
            descriptor: descriptor to assert
        """
        binaryDesc = descriptor.asBytes
        self.assertTrue(0.0 <= descriptor.garbageScore <= 1.0, descriptor.garbageScore)
        self.assertEqual(self.version, descriptor.model)
        self.assertEqual(list(binaryDesc), descriptor.asVector)
        self.assertEqual(b"dp\x00\x00" + self.version.to_bytes(4, "little") + binaryDesc, descriptor.rawDescriptor)

    def test_descriptor_methods(self):
        """
        Test a descriptor.
        """
        self.assertDescriptor(self.descriptor)

    def test_replaced_descriptor_methods(self):
        """
        Test a replaced descriptor.
        """
        gs = 0.777
        rawBinaryDesc = self.descriptor.rawDescriptor
        binaryDesc = rawBinaryDesc[8:]
        descriptor: FaceDescriptor = self._estimator.descriptorFactory.generateDescriptor()

        descriptor.reload(rawBinaryDesc, garbageScore=gs)

        self.assertDescriptor(descriptor)
        self.assertEqual(binaryDesc, descriptor.asBytes)
        self.assertEqual(rawBinaryDesc, descriptor.rawDescriptor)
        self.assertEqual(gs, descriptor.garbageScore)

    def test_aggregated_descriptor_methods(self):
        """
        Test aggregated method.
        """
        self.assertDescriptor(self.aggregatedDescriptor)

    def test_descriptor_batch_methods(self):
        """
        Test descriptor batch methods.
        """
        with self.subTest("__getitem__"):
            for idx in range(len(warps)):
                descr = self.descriptorBatch[idx]
                self.assertDescriptor(descr)

        with self.subTest("__len__"):
            self.assertEqual(len(warps), len(self.descriptorBatch))

        with self.subTest("__iter__"):
            descriptors = list(self.descriptorBatch)
            self.assertEqual(len(warps), len(descriptors))

    def test_descriptor_batch_methods_bad(self):
        """
        Test descriptor batch methods bad.
        """
        maxLength = 2
        descriptorBatch = self._estimator.descriptorFactory.generateDescriptorsBatch(maxLength, self.version)
        descriptorBatch.append(self.descriptor)
        for name, idx in (("empty", 1), ("nonexistent", 3)):
            with self.subTest(name=name):
                with self.assertRaises(IndexError) as e:
                    descriptorBatch[idx]
                self.assertIn(str(idx), e.exception.args[0])

    def test_descriptor_batch_append(self):
        """
        Test descriptor batch append.
        """
        maxLength = 3
        descriptorBatch = self._estimator.descriptorFactory.generateDescriptorsBatch(maxLength, self.version)
        self.assertEqual(0, len(descriptorBatch))
        for idx in range(maxLength):
            with self.subTest(idx):
                descriptorBatch.append(self.aggregatedDescriptor)
                self.assertEqual(idx + 1, len(descriptorBatch))


class TestEstimateDescriptor(BaseTestClass):
    """
    Test estimate descriptor.
    """

    def assertDescriptor(self, expectedVersion: int, descriptor: FaceDescriptor) -> None:
        """
        Assert extracted descriptor.

        Args:
            expectedVersion: expected descriptor version
            descriptor: extracted descriptor
        """
        self.assertIsInstance(descriptor, FaceDescriptor)
        self.assertEqual(descriptor.model, expectedVersion, "descriptor has wrong version")
        length = {46: 256, 52: 256, 54: 512, 56: 512}[expectedVersion]
        self.assertEqual(length, len(descriptor.asBytes))
        self.assertEqual(length, len(descriptor.asVector))
        self.assertEqual(length + 8, len(descriptor.rawDescriptor))

    def getDescr(self, planVersion) -> FaceDescriptor:
        """
        Get some descriptor.

        Args:
            planVersion: version of descriptor

        Returns:
            descriptor
        """
        return self.faceEngine.createFaceDescriptorFactory(planVersion).generateDescriptor()

    def getBatch(self, planVersion, size) -> FaceDescriptorBatch:
        """
        Get some descriptor batch.

        Args:
            planVersion: version of descriptor batch
            size: number of descriptors in batch

        Returns:
            descriptor
        """
        return self.faceEngine.createFaceDescriptorFactory(planVersion).generateDescriptorsBatch(size)

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

    @pytest.mark.skip("need support 57 version")
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

    def test_extract_descriptors_positive(self):
        """
        Test correctly estimate descriptor.
        """
        for planVersion in EDVa:
            for kw in (dict(), dict(descriptor=self.getDescr(planVersion))):
                extractor = self.faceEngine.createFaceDescriptorEstimator(descriptorVersion=planVersion)
                with self.subTest(plan_version=planVersion, external_descriptor=bool(kw)):
                    descriptor = extractor.estimate(warp, **kw)
                    self.assertDescriptor(planVersion, descriptor)

    @unittest.skip("dont do it FSDK-2186")
    def test_extract_descriptors_incorrect_source_descriptors(self):
        """
        Test estimate descriptor using incorrect source descriptor.
        """
        for planVersion in EDVa:
            extractor = self.faceEngine.createFaceDescriptorEstimator(descriptorVersion=planVersion)
            for descriptorVersion in set(EDVa) - {planVersion}:
                descriptorOfAnotherVersion = self.getDescr(descriptorVersion)
                with self.subTest(plan_version=planVersion, descriptor_version=descriptorVersion):
                    print(f"Plan {planVersion}, empty descriptor version {descriptorVersion}")
                    descriptor = extractor.estimate(warp, descriptor=descriptorOfAnotherVersion)
                    self.assertDescriptor(planVersion, descriptor)

    def test_extract_descriptors_batch_positive(self):
        """
        Test correctly estimate descriptor batch.
        """
        for planVersion in EDVa:
            extractor = self.faceEngine.createFaceDescriptorEstimator(descriptorVersion=planVersion)
            for kw in (dict(), dict(descriptorBatch=self.getBatch(planVersion, len(warps)))):
                for aggregate in (True, False):
                    with self.subTest(plan_version=planVersion, aggregate=aggregate, external_descriptor=bool(kw)):
                        descriptorsRaw, descriptorAggregated = extractor.estimateDescriptorsBatch(
                            warps, aggregate=aggregate, **kw
                        )
                        for descriptorRaw in descriptorsRaw:
                            self.assertDescriptor(planVersion, descriptorRaw)

                        if aggregate:
                            self.assertDescriptor(planVersion, descriptorAggregated)
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
                for kw in (dict(), dict(descriptorBatch=self.getBatch(descriptorVersion, len(warps)))):
                    for aggregate in (True, False):
                        with self.subTest(plan_version=planVersion, aggregate=aggregate, external_descriptor=bool(kw)):
                            descriptorsRaw, descriptorAggregated = extractor.estimateDescriptorsBatch(
                                warps, aggregate=aggregate, **kw
                            )
                            for descriptorRaw in descriptorsRaw:
                                self.assertDescriptor(planVersion, descriptorRaw)

                            if aggregate:
                                self.assertDescriptor(planVersion, descriptorAggregated)
                            else:
                                self.assertIsNone(descriptorAggregated)
