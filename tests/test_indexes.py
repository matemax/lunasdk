"""
Test build an index with descriptors.
"""
import pytest

from lunavl.sdk.descriptors.descriptors import (
    FaceDescriptor,
    FaceDescriptorBatch,
)
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.indexes.base import IndexBuilder
from lunavl.sdk.indexes.dynamic_index import DynamicIndex, IndexResult, IndexType
from tests.base import BaseTestClass
from tests.resources import WARP_WHITE_MAN, WARP_ONE_FACE, WARP_CLEAN_FACE

faceWarp = FaceWarpedImage.load(filename=WARP_WHITE_MAN)
faceWarps = [FaceWarpedImage.load(filename=WARP_CLEAN_FACE), FaceWarpedImage.load(filename=WARP_ONE_FACE)]


class TestIndexFunctionality(BaseTestClass):
    """Test of indexes."""

    faceDescriptor: FaceDescriptor
    nonDefaultFaceDescriptor: FaceDescriptor
    faceDescriptorBatch: FaceDescriptorBatch
    indexBuilder: IndexBuilder

    @classmethod
    def setup_class(cls):
        super().setup_class()
        defaultEstimator = BaseTestClass.faceEngine.createFaceDescriptorEstimator(54)
        cls.faceDescriptor = defaultEstimator.estimate(faceWarp)
        cls.faceDescriptorBatch, _ = defaultEstimator.estimateDescriptorsBatch(faceWarps)

        otherEstimator = BaseTestClass.faceEngine.createFaceDescriptorEstimator(56)
        cls.nonDefaultFaceDescriptor = otherEstimator.estimate(faceWarp)

    def setUp(self) -> None:
        super().setUp()
        self.indexBuilder = self.faceEngine.createIndexBuilder()

    @staticmethod
    def assertDynamicIndex(dynamicIndex: DynamicIndex, expectedDescriptorCount: int, expectedBufSize: int):
        """
        Assert instance class for dynamic index and check storage size.
        Args:
            dynamicIndex: input index
            expectedDescriptorCount: expected count of descriptors
            expectedBufSize: expected index buffer size
        """
        assert isinstance(dynamicIndex, DynamicIndex), f"created {dynamicIndex} is not {DynamicIndex}"
        assert isinstance(dynamicIndex.bufSize, int), f"expected int but found {dynamicIndex.bufSize}"
        assert expectedBufSize == dynamicIndex.bufSize, "dynamic buf size is not equal to the expected"
        assert isinstance(dynamicIndex.count, int), f"expected int but found {dynamicIndex.count}"
        assert expectedDescriptorCount == dynamicIndex.count, "count of descriptors is not equal to the expected"

    def test_build_empty_index(self):
        """Test build empty index."""
        index = self.indexBuilder.buildIndex()
        assert 0 == self.indexBuilder.bufSize
        self.assertDynamicIndex(index, 0, 0)

    def test_append_descriptor_to_builder(self):
        """Test append descriptor to index builder."""
        self.indexBuilder.append(self.faceDescriptor)
        assert 1 == self.indexBuilder.bufSize
        index = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(index, 1, 1)

    def test_append_descriptors_batch_to_builder(self):
        """Test append descriptors batch to index builder."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        assert 2 == self.indexBuilder.bufSize
        index = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(index, 2, 2)

    def test_get_descriptor_from_builder(self):
        """Test get descriptor from internal storage."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        assert 2 == self.indexBuilder.bufSize
        for idx in range(len(self.faceDescriptorBatch)):
            with self.subTest(case=f"get descriptor with index: {idx}"):
                descriptor = self.indexBuilder.getDescriptor(idx, self.faceDescriptor)
                assert self.faceDescriptorBatch[idx].rawDescriptor == descriptor.rawDescriptor

    def test_get_descriptor_from_builder_bad_index(self):
        """Test get descriptor with invalid index from internal storage."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        assert 2 == self.indexBuilder.bufSize
        nonexistentIndex = 2
        with self.assertRaises(IndexError) as ex:
            self.indexBuilder.getDescriptor(nonexistentIndex, self.faceDescriptor)
        assert str(nonexistentIndex) in ex.exception.args[0]

    @pytest.mark.skip("FSDK get descriptor error: version mismatch")
    def test_get_non_default_descriptor_from_builder(self):
        """Test get non default descriptor from internal storage."""
        self.indexBuilder.append(self.faceDescriptor)
        assert 1 == self.indexBuilder.bufSize
        with pytest.raises(LunaSDKException) as ex:
            self.indexBuilder.getDescriptor(0, self.nonDefaultFaceDescriptor)
        self.assertLunaVlError(ex, LunaVLError.InvalidInput)

    def test_remove_descriptor_from_builder(self):
        """Test remove descriptor from internal storage."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        assert 2 == self.indexBuilder.bufSize
        for index, expectedBufSize in ((1, 1), (0, 0)):
            with self.subTest(case=f"remove descriptor with index: {index}"):
                del self.indexBuilder[index]
                assert expectedBufSize == self.indexBuilder.bufSize

    def test_remove_descriptor_from_builder_bad(self):
        """Test remove descriptor with invalid index from internal storage."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        assert 2 == self.indexBuilder.bufSize
        nonexistentIndex = 2
        with self.assertRaises(IndexError) as ex:
            del self.indexBuilder[nonexistentIndex]
        assert str(nonexistentIndex) in ex.exception.args[0]

    @pytest.mark.skip("Segmentation fault")
    def test_append_descriptor_to_empty_dynamic_index(self):
        """Test append descriptor to empty dynamic index."""
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 0, 0)
        dynamicIndex.append(self.faceDescriptor)
        self.assertDynamicIndex(dynamicIndex, 1, 1)

    def test_append_descriptor_to_dynamic_index(self):
        """Test append descriptor to dynamic index."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 1, 1)
        dynamicIndex.append(self.faceDescriptor)
        self.assertDynamicIndex(dynamicIndex, 2, 2)

    def test_append_descriptors_batch_to_dynamic_index(self):
        """Test append descriptor batch to dynamic index."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 1, 1)
        dynamicIndex.appendBatch(self.faceDescriptorBatch)
        self.assertDynamicIndex(dynamicIndex, 3, 3)

    def test_get_descriptor_from_dynamic_index(self):
        """Test get descriptor from dynamic index."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 2, 2)
        for idx in range(len(self.faceDescriptorBatch)):
            with self.subTest(case=f"get descriptor with index: {idx}"):
                descriptor = dynamicIndex.getDescriptor(idx, self.faceDescriptor)
                assert self.faceDescriptorBatch[idx].rawDescriptor == descriptor.rawDescriptor

    def test_remove_descriptor_from_dynamic_index(self):
        """Test remove descriptor from dynamic index."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 2, 2)
        for index, expectedDescriptorCount in ((1, 1), (0, 0)):
            with self.subTest(case=f"remove descriptor with index: {index}"):
                del dynamicIndex[0]
                self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount, len(self.faceDescriptorBatch))

    def test_search_similar_descriptor(self):
        """Test search result for descriptors."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 2, 2)
        result = dynamicIndex.search(self.faceDescriptor, len(self.faceDescriptorBatch))
        assert 2 == len(result), result
        assert result[0].distance < result[1].distance, "first result should be the minimum distance"
        topResult = result[0]
        assert isinstance(topResult, IndexResult), f"result {topResult} is not {IndexResult}"
        assert isinstance(topResult.distance, float), f"expected float but found {topResult.distance}"
        assert isinstance(topResult.similarity, float), f"expected float but found {topResult.similarity}"
        assert isinstance(topResult.index, int), f"expected int but found {topResult.index}"
        assert len(self.faceDescriptorBatch) > topResult.index, "dynamic index out of range"
        assert 0.0 < topResult.similarity < 1.0, "similarity out of range [0,1]"
