"""
Test estimate descriptor.
"""
import unittest
from enum import Enum
from typing import Tuple, Generator, NamedTuple, Callable, List, Union, ContextManager

import pytest

from lunavl.sdk.descriptors.descriptors import (
    FaceDescriptor,
    BaseDescriptor,
    BaseDescriptorBatch,
    HumanDescriptor,
    FaceDescriptorBatch,
    HumanDescriptorBatch,
)
from lunavl.sdk.estimators.body_estimators.human_descriptor import HumanDescriptorEstimator
from lunavl.sdk.estimators.body_estimators.humanwarper import HumanWarpedImage
from lunavl.sdk.estimators.face_estimators.face_descriptor import FaceDescriptorEstimator
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.globals import DEFAULT_HUMAN_DESCRIPTOR_VERSION as DHDV
from tests.base import BaseTestClass
from tests.resources import WARP_WHITE_MAN, HUMAN_WARP


EFDVa = EXISTENT_FACE_DESCRIPTOR_VERSION_ABUNDANCE = [46, 52, 54, 56]

EHDVa = EXISTENT_HUMAN_DESCRIPTOR_VERSION_ABUNDANCE = [DHDV]


class DescriptorType(Enum):
    face = "face"
    human = "human"


faceWarp = FaceWarpedImage.load(filename=WARP_WHITE_MAN)
faceWarps = [faceWarp] * 3
humanWarp = HumanWarpedImage.load(filename=HUMAN_WARP)
humanWarps = [humanWarp] * 3


class DescriptorCase(NamedTuple):
    descriptor: BaseDescriptor
    aggregatedDescriptor: BaseDescriptor
    descriptorBatch: BaseDescriptorBatch
    type: DescriptorType
    estimator: Union[FaceDescriptorEstimator, HumanDescriptorEstimator]


class ExtractorCase(NamedTuple):
    type: DescriptorType
    extractorFactory: Callable
    versions: List[int]
    warps: Union[List[HumanWarpedImage], List[FaceWarpedImage]]


class TestDescriptorFunctionality(BaseTestClass):
    faceDescriptorVersion: int = EFDVa[0]
    faceEstimator: FaceDescriptorEstimator
    faceDescriptor: FaceDescriptor
    faceDescriptorBatch: FaceDescriptorBatch
    aggregatedFaceDescriptor: FaceDescriptor

    humanDescriptorVersion: int = EHDVa[0]
    humanEstimator: HumanDescriptorEstimator
    humanDescriptor: HumanDescriptor
    humanDescriptorBatch: HumanDescriptorBatch
    aggregatedHumanDescriptor: HumanDescriptor

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.faceEstimator = BaseTestClass.faceEngine.createFaceDescriptorEstimator(cls.faceDescriptorVersion)
        cls.faceDescriptor = cls.faceEstimator.estimate(faceWarp)
        estimator = cls.faceEstimator
        cls.faceDescriptorBatch, cls.aggregatedFaceDescriptor = estimator.estimateDescriptorsBatch(
            faceWarps, aggregate=True
        )

        cls.humanDescriptorVersion = EHDVa[0]
        estimator = BaseTestClass.faceEngine.createHumanDescriptorEstimator(cls.humanDescriptorVersion)
        cls.humanEstimator = estimator
        cls.humanDescriptor = cls.humanEstimator.estimate(humanWarp)
        cls.humanDescriptorBatch, cls.aggregatedHumanDescriptor = estimator.estimateDescriptorsBatch(
            humanWarps, aggregate=True
        )

    def descriptorSubTest(
        self,
    ) -> Generator[Tuple[ContextManager[None], DescriptorCase], Tuple[ContextManager[None], DescriptorCase], None]:
        """
        Generator for sub tests for human descriptor and face descriptor.
        """
        for descriptorType in DescriptorType:

            subTest = self.subTest(descriptorType=descriptorType)
            if descriptorType == DescriptorType.human:
                descriptor: BaseDescriptor = self.humanDescriptor
                aggregatedDescriptor: BaseDescriptor = self.aggregatedHumanDescriptor
                descriptorBatch: BaseDescriptorBatch = self.humanDescriptorBatch
                estimator: Union[HumanDescriptorEstimator, FaceDescriptorEstimator] = self.humanEstimator
            else:
                descriptor = self.faceDescriptor
                aggregatedDescriptor = self.aggregatedFaceDescriptor
                descriptorBatch = self.faceDescriptorBatch
                estimator = self.faceEstimator
            yield subTest, DescriptorCase(descriptor, aggregatedDescriptor, descriptorBatch, descriptorType, estimator)

    def assertDescriptor(self, descriptor: BaseDescriptor, descriptorType: DescriptorType):
        """
        Assert descriptor format.

        Args:
            descriptor: descriptor to assert
            descriptorType: descriptor Type
        """
        binaryDesc = descriptor.asBytes
        assert 0.0 <= descriptor.garbageScore <= 1.0, descriptor.garbageScore
        assert list(binaryDesc) == descriptor.asVector
        if descriptorType == DescriptorType.face:
            version = self.faceDescriptorVersion
        else:
            version = self.humanDescriptorVersion

        assert version == descriptor.model
        assert b"dp\x00\x00" + version.to_bytes(4, "little") + binaryDesc == descriptor.rawDescriptor

    def test_descriptor_methods(self):
        """
        Test a descriptor.
        """
        for subTest, case in self.descriptorSubTest():
            with subTest:
                self.assertDescriptor(case.descriptor, case.type)

    def test_replaced_descriptor_methods(self):
        """
        Test a replaced descriptor.
        """
        for subTest, case in self.descriptorSubTest():
            with subTest:
                gs = 0.777
                rawBinaryDesc = case.descriptor.rawDescriptor
                binaryDesc = rawBinaryDesc[8:]
                descriptor = case.estimator.descriptorFactory.generateDescriptor()
                descriptor.reload(rawBinaryDesc, garbageScore=gs)

                self.assertDescriptor(descriptor, case.type)
                assert binaryDesc == descriptor.asBytes
                assert rawBinaryDesc == descriptor.rawDescriptor
                assert gs == descriptor.garbageScore

    def test_aggregated_descriptor_methods(self):
        """
        Test aggregated method.
        """
        for subTest, case in self.descriptorSubTest():
            with subTest:
                self.assertDescriptor(case.aggregatedDescriptor, case.type)

    def test_descriptor_batch_methods(self):
        """
        Test descriptor batch methods.
        """
        for subTest, case in self.descriptorSubTest():
            with subTest:
                with self.subTest("__getitem__"):
                    for idx in range(len(faceWarps)):
                        descr = case.descriptorBatch[idx]
                        self.assertDescriptor(descr, case.type)

                if case.type == DescriptorType.face:
                    sourceWarp = faceWarps
                else:
                    sourceWarp = humanWarps

                assert len(sourceWarp) == len(case.descriptorBatch)

                descriptors = list(case.descriptorBatch)
                assert len(sourceWarp) == len(descriptors)

    def test_descriptor_batch_methods_bad(self):
        """
        Test descriptor batch methods bad.
        """
        for subTest, case in self.descriptorSubTest():
            with subTest:

                maxLength = 2

                descriptorFactory = case.estimator.descriptorFactory
                if case.type == DescriptorType.face:
                    descriptorBatch = descriptorFactory.generateDescriptorsBatch(maxLength, self.faceDescriptorVersion)
                else:
                    descriptorBatch = descriptorFactory.generateDescriptorsBatch(maxLength)

                descriptorBatch.append(case.descriptor)
                for name, idx in (("empty", 1), ("nonexistent", 3)):
                    with self.subTest(name=name):
                        with self.assertRaises(IndexError) as e:
                            descriptorBatch[idx]
                        assert str(idx) in e.exception.args[0]

    def test_descriptor_batch_append(self):
        """
        Test descriptor batch append.
        """
        for subTest, case in self.descriptorSubTest():
            with subTest:

                maxLength = 3
                descriptorFactory = case.estimator.descriptorFactory
                if case.type == DescriptorType.face:
                    descriptorBatch = descriptorFactory.generateDescriptorsBatch(maxLength, self.faceDescriptorVersion)
                else:
                    descriptorBatch = descriptorFactory.generateDescriptorsBatch(maxLength)

                assert 0 == len(descriptorBatch)
                for idx in range(maxLength):
                    with self.subTest(idx):
                        descriptorBatch.append(case.aggregatedDescriptor)
                        assert idx + 1 == len(descriptorBatch)


class TestEstimateDescriptor(BaseTestClass):
    """
    Test estimate descriptor.
    """

    cases: Tuple[ExtractorCase, ExtractorCase]

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.cases = (
            ExtractorCase(DescriptorType.face, cls.faceEngine.createFaceDescriptorEstimator, EFDVa, faceWarps),
            ExtractorCase(DescriptorType.human, cls.faceEngine.createHumanDescriptorEstimator, EHDVa, humanWarps),
        )

    @staticmethod
    def assertDescriptor(expectedVersion: int, descriptor: BaseDescriptor, descriptorType: DescriptorType) -> None:
        """
        Assert extracted descriptor.

        Args:
            expectedVersion: expected descriptor version
            descriptor: extracted descriptor
            descriptorType: descriptor type
        """
        if descriptorType == DescriptorType.face:
            assert isinstance(descriptor, FaceDescriptor)
        else:
            assert isinstance(descriptor, HumanDescriptor)
        assert descriptor.model == expectedVersion, "descriptor has wrong version"
        length = {46: 256, 52: 256, 54: 512, 56: 512, 101: 2048}[expectedVersion]
        assert length == len(descriptor.asBytes)
        assert length == len(descriptor.asVector)
        assert length + 8 == len(descriptor.rawDescriptor)

    def getDescr(self, planVersion, descriptorType: DescriptorType) -> BaseDescriptor:
        """
        Get some descriptor.

        Args:
            planVersion: version of descriptor
            descriptorType: descriptor type
        Returns:
            descriptor
        """
        if descriptorType == DescriptorType.face:
            return self.faceEngine.createFaceDescriptorFactory(planVersion).generateDescriptor()
        else:
            return self.faceEngine.createHumanDescriptorFactory(planVersion).generateDescriptor()

    def getBatch(self, planVersion, size, descriptorType: DescriptorType) -> BaseDescriptorBatch:
        """
        Get some descriptor batch.

        Args:
            planVersion: version of descriptor batch
            size: number of descriptors in batch
            descriptorType: descriptor type
        Returns:
            descriptor
        """
        if descriptorType == DescriptorType.face:
            return self.faceEngine.createFaceDescriptorFactory(planVersion).generateDescriptorsBatch(size)
        else:
            return self.faceEngine.createHumanDescriptorFactory(planVersion).generateDescriptorsBatch(size)

    def test_create_estimators_positive(self):
        """
        Test create estimators of different existent plan versions.
        """
        for case in self.cases:
            with self.subTest(type=case.type):
                for planVersion in case.versions:

                    with self.subTest(descriptor_version=planVersion):
                        try:
                            case.extractorFactory(descriptorVersion=planVersion)
                        except RuntimeError as e:
                            raise AssertionError(
                                f"Descriptor version {planVersion} is not supported. But must be."
                            ) from e

    @pytest.mark.skip("need support 57 version")
    def test_create_estimators_negative(self):
        """
        Test create estimators of different nonexistent plan versions.
        """
        for case in self.cases:
            with self.subTest(type=case.type):

                nonexistentVersions = set(range(min(case.versions) - 10, max(case.versions) + 10)) - set(case.versions)
                for planVersion in nonexistentVersions:
                    with self.subTest(descriptor_version=planVersion):
                        try:
                            case.extractorFactory(descriptorVersion=planVersion)
                        except RuntimeError:
                            pass
                        else:
                            raise AssertionError(f"Descriptor version {planVersion} is supported. But should not.")

    def test_extract_descriptors_positive(self):
        """
        Test correctly estimate descriptor.
        """
        for case in self.cases:
            with self.subTest(type=case.type):

                for planVersion in case.versions:
                    for kw in (dict(), dict(descriptor=self.getDescr(planVersion, case.type))):
                        extractor = case.extractorFactory(descriptorVersion=planVersion)
                        with self.subTest(plan_version=planVersion, external_descriptor=bool(kw)):
                            descriptor = extractor.estimate(case.warps[0], **kw)
                            self.assertDescriptor(planVersion, descriptor, case.type)

    @unittest.skip("dont do it FSDK-2186")
    def test_extract_descriptors_incorrect_source_descriptors(self):
        """
        Test estimate descriptor using incorrect source descriptor.
        """
        for case in self.cases:
            with self.subTest(type=case.type):

                for planVersion in case.versions:
                    extractor = case.extractorFactory(descriptorVersion=planVersion)
                    for descriptorVersion in set(EFDVa) - {planVersion}:
                        descriptorOfAnotherVersion = self.getDescr(descriptorVersion, case.type)
                        with self.subTest(plan_version=planVersion, descriptor_version=descriptorVersion):
                            print(f"Plan {planVersion}, empty descriptor version {descriptorVersion}")
                            descriptor = extractor.estimate(case.warps[0], descriptor=descriptorOfAnotherVersion)
                            self.assertDescriptor(planVersion, descriptor, case.type)

    def test_extract_descriptors_batch_positive(self):
        """
        Test correctly estimate descriptor batch.
        """
        for case in self.cases:
            with self.subTest(type=case.type):

                for planVersion in case.versions:
                    extractor = case.extractorFactory(descriptorVersion=planVersion)
                    for kw in (dict(), dict(descriptorBatch=self.getBatch(planVersion, len(case.warps), case.type))):
                        for aggregate in (True, False):
                            with self.subTest(
                                plan_version=planVersion, aggregate=aggregate, external_descriptor=bool(kw)
                            ):
                                descriptorsRaw, descriptorAggregated = extractor.estimateDescriptorsBatch(
                                    case.warps, aggregate=aggregate, **kw
                                )
                                for descriptorRaw in descriptorsRaw:
                                    self.assertDescriptor(planVersion, descriptorRaw, case.type)

                                if aggregate:
                                    self.assertDescriptor(planVersion, descriptorAggregated, case.type)
                                else:
                                    assert descriptorAggregated is None

    @unittest.skip("dont do it FSDK-2186")
    def test_extract_descriptors_batch_incorrect_source_descriptors(self):
        """
        Test correctly estimate descriptor batch.
        """
        for case in self.cases:
            with self.subTest(type=case.type):

                for planVersion in case.versions:
                    extractor = case.extractorFactory(descriptorVersion=planVersion)
                    for descriptorVersion in set(EFDVa) - {planVersion}:
                        for kw in (
                            dict(),
                            dict(descriptorBatch=self.getBatch(descriptorVersion, len(case.warps), case.type)),
                        ):
                            for aggregate in (True, False):
                                with self.subTest(
                                    plan_version=planVersion, aggregate=aggregate, external_descriptor=bool(kw)
                                ):
                                    descriptorsRaw, descriptorAggregated = extractor.estimateDescriptorsBatch(
                                        case.warps, aggregate=aggregate, **kw
                                    )
                                    for descriptorRaw in descriptorsRaw:
                                        self.assertDescriptor(planVersion, descriptorRaw, case.type)

                                    if aggregate:
                                        self.assertDescriptor(planVersion, descriptorAggregated, case.type)
                                    else:
                                        assert descriptorAggregated is not None
