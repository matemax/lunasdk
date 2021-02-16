"""
Test build an index with descriptors.
"""
from typing import Union

import pytest
import os
from lunavl.sdk.descriptors.descriptors import (
    FaceDescriptor,
    FaceDescriptorBatch,
)
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage
from lunavl.sdk.indexes.base import IndexResult
from lunavl.sdk.indexes.builder import IndexBuilder
from lunavl.sdk.indexes.stored_index import DynamicIndex, DenseIndex, IndexType
from lunavl.sdk.estimators.face_estimators.face_descriptor import FaceDescriptorEstimator
from tests.base import BaseTestClass
from tests.resources import WARP_WHITE_MAN, WARP_ONE_FACE, WARP_CLEAN_FACE

faceWarp = FaceWarpedImage.load(filename=WARP_WHITE_MAN)
faceWarps = [FaceWarpedImage.load(filename=WARP_CLEAN_FACE), FaceWarpedImage.load(filename=WARP_ONE_FACE)]
currDir = os.path.dirname(__file__)
pathToStoredIndex = os.path.join(currDir, "stored.index")


class TestIndexFunctionality(BaseTestClass):
    """Test of indexes."""

    defaultFaceEstimator: FaceDescriptorEstimator
    faceDescriptor: FaceDescriptor
    nonDefaultFaceDescriptor: FaceDescriptor
    faceDescriptorBatch: FaceDescriptorBatch
    indexBuilder: IndexBuilder

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.defaultFaceEstimator = BaseTestClass.faceEngine.createFaceDescriptorEstimator(54)

        nonDefaultEstimator = BaseTestClass.faceEngine.createFaceDescriptorEstimator(56)
        cls.nonDefaultFaceDescriptor = nonDefaultEstimator.estimate(faceWarp)

    def setUp(self) -> None:
        super().setUp()
        self.indexBuilder = self.faceEngine.createFaceIndex()
        self.faceDescriptor = self.defaultFaceEstimator.estimate(faceWarp)
        self.faceDescriptorBatch, _ = self.defaultFaceEstimator.estimateDescriptorsBatch(faceWarps)

    def tearDown(self) -> None:
        super().tearDown()
        if os.path.exists(pathToStoredIndex):
            os.remove(pathToStoredIndex)

    def assertDynamicIndex(self, dynamicIndex: DynamicIndex, expectedDescriptorCount: int, expectedBufSize: int):
        """
        Assert instance class for dynamic index and check storage size.
        Args:
            dynamicIndex: input index
            expectedDescriptorCount: expected count of descriptors
            expectedBufSize: expected index buffer size
        """
        assert isinstance(dynamicIndex, DynamicIndex), f"created {dynamicIndex} is not {DynamicIndex}"
        assert isinstance(dynamicIndex.bufSize, int), f"expected int but found {dynamicIndex.bufSize}"
        assert isinstance(dynamicIndex.descriptorsCount, int), f"expected int but found {dynamicIndex.descriptorsCount}"
        assert expectedBufSize == dynamicIndex.bufSize, "dynamic buf size is not equal to the expected"
        assert expectedDescriptorCount == dynamicIndex.descriptorsCount, \
            "count of descriptors is not equal to the expected"
        assert self.getCountOfDescriptorsInStorage(dynamicIndex) == dynamicIndex.bufSize, \
            "wrong size of internal storage"

    @staticmethod
    def getCountOfDescriptorsInStorage(indexObject: Union[IndexBuilder, DynamicIndex, DenseIndex]) -> int:
        """
        Get actual count of descriptor in internal storage.
        Args:
            indexObject: an initialized index object with `getDescriptor` method
        Returns:
            count of descriptors
        """
        count = 0
        while True:
            try:
                indexObject[count]
            except IndexError:
                break
            else:
                count += 1
        return count

    def test_build_empty_index(self):
        """Test build empty index."""
        expectedDescriptorsCount = 0
        index = self.indexBuilder.buildIndex()
        assert expectedDescriptorsCount == self.indexBuilder.bufSize
        self.assertDynamicIndex(index, expectedDescriptorCount=expectedDescriptorsCount, expectedBufSize=0)

    def test_append_descriptor_to_builder(self):
        """Test append descriptor to index builder."""
        expectedDescriptorsCount = 1
        self.indexBuilder.append(self.faceDescriptor)
        assert expectedDescriptorsCount == self.indexBuilder.bufSize
        assert expectedDescriptorsCount == self.getCountOfDescriptorsInStorage(self.indexBuilder)

    def test_append_descriptors_batch_to_builder(self):
        """Test append descriptors batch to index builder."""
        expectedDescriptorsCount = 2
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        assert expectedDescriptorsCount == self.indexBuilder.bufSize
        assert expectedDescriptorsCount == self.getCountOfDescriptorsInStorage(self.indexBuilder)

    def test_get_descriptor_from_builder(self):
        """Test get descriptor from internal storage."""
        expectedDescriptorsCount = 2
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        assert expectedDescriptorsCount == self.indexBuilder.bufSize
        for idx in range(expectedDescriptorsCount):
            with self.subTest(case=f"get descriptor with index: {idx}"):
                descriptor = self.indexBuilder[idx]
                assert self.faceDescriptorBatch[idx].rawDescriptor == descriptor.rawDescriptor

    def test_get_descriptor_from_builder_bad_index(self):
        """Test get descriptor with invalid index from internal storage."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        assert len(self.faceDescriptorBatch) == self.indexBuilder.bufSize
        nonexistentIndex = 2
        with self.assertRaises(IndexError) as ex:
            descriptor = self.indexBuilder[nonexistentIndex]
        assert str(nonexistentIndex) in ex.exception.args[0]

    def test_get_non_default_descriptor_from_builder(self):
        """Test get non default descriptor from internal storage."""
        indexBuilder = self.faceEngine.createFaceIndex(descriptorVersion=56)
        indexBuilder.append(self.faceDescriptor)
        assert 1 == indexBuilder.bufSize
        with pytest.raises(LunaSDKException) as ex:
            descriptor = indexBuilder[0]
        self.assertLunaVlError(ex, LunaVLError.InvalidInput.format("Invalid input"))

    def test_remove_descriptor_from_builder(self):
        """Test remove descriptor from internal storage."""
        expectedDescriptorsCount = 2
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        assert expectedDescriptorsCount == self.indexBuilder.bufSize
        for index, expectedBufSize in ((1, 1), (0, 0)):
            with self.subTest(case=f"remove descriptor with index: {index}"):
                del self.indexBuilder[index]
                assert expectedBufSize == self.indexBuilder.bufSize

    def test_remove_descriptor_from_builder_bad(self):
        """Test remove descriptor with invalid index from internal storage."""
        expectedDescriptorsCount = 2
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        assert expectedDescriptorsCount == self.indexBuilder.bufSize
        nonexistentIndex = 2
        with self.assertRaises(IndexError) as ex:
            del self.indexBuilder[nonexistentIndex]
        assert str(nonexistentIndex) in ex.exception.args[0]

    @pytest.mark.skip("FSDK-2877 Segmentation fault")
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
                descriptor = dynamicIndex[idx]
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
        assert 0.0 < topResult.similarity <= 1.0, "similarity out of range [0,1]"

    def test_search_result_non_default_descriptor(self):
        """Test search result by descriptor different version."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 2, 2)
        with pytest.raises(LunaSDKException) as ex:
            dynamicIndex.search(self.nonDefaultFaceDescriptor)
        self.assertLunaVlError(ex, LunaVLError.InvalidInput.format("Invalid input"))

    @pytest.mark.skip("FSDK-2877 internal error")
    def test_search_result_empty(self):
        """Test search with empty result."""
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 0, 0)
        result = dynamicIndex.search(self.faceDescriptor)
        assert [] == result

    def test_search_result_invalid_input(self):
        """Test search with invalid parameter."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 1, 1)
        with pytest.raises(LunaSDKException) as ex:
            dynamicIndex.search(self.faceDescriptor, maxCount=0)
        self.assertLunaVlError(ex, LunaVLError.InvalidInput.format("Invalid input"))

    def test_save_index_as_dynamic(self):
        """Test save index as dynamic."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 1, 1)

        dynamicIndex.save(pathToStoredIndex, IndexType.dynamic)
        assert os.path.isfile(pathToStoredIndex), "dynamic index file not found"
        dynamicIndex = self.indexBuilder.loadIndex(pathToStoredIndex, IndexType.dynamic)
        self.assertDynamicIndex(dynamicIndex, 1, 1)

    def test_save_index_as_dense(self):
        """Test save index as dense."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 1, 1)

        dynamicIndex.save(pathToStoredIndex, IndexType.dense)
        assert os.path.isfile(pathToStoredIndex), "dense index file not found"
        denseIndex = self.indexBuilder.loadIndex(pathToStoredIndex, IndexType.dense)
        assert isinstance(denseIndex, DenseIndex), f"created {denseIndex} is not {DenseIndex}"
        assert isinstance(denseIndex.bufSize, int), f"expected int but found {denseIndex.bufSize}"
        assert 1 == dynamicIndex.bufSize, "dense buf size is not equal to the expected"
        assert self.getCountOfDescriptorsInStorage(denseIndex) == denseIndex.bufSize, \
            "wrong size of internal storage"

    def test_save_index_bad_filename(self):
        """Test save index to local storage with bad filename."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, 1, 1)

        for path in (os.path.join(currDir, ".."), os.path.join(currDir, "./"), currDir):
            with self.subTest(path=path):
                with self.assertRaises(ValueError) as ex:
                    dynamicIndex.save(path, IndexType.dynamic)
                assert f"{path} must not be a directory" in ex.exception.args[0]

    @pytest.mark.skip("FSDK RuntimeError: Failed to read index metadata")
    def test_load_index_unknown_file(self):
        """Test load index unknown file."""
        with pytest.raises(LunaSDKException) as ex:
            self.faceEngine.loadDynamicIndex(WARP_ONE_FACE)
        self.assertLunaVlError(ex, LunaVLError.InvalidInput)
