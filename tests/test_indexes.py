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

EFDVa = EXISTENT_FACE_DESCRIPTOR_VERSION_ABUNDANCE = [54, 56, 57, 58]

faceWarp = FaceWarpedImage.load(filename=WARP_WHITE_MAN)
faceWarps = [FaceWarpedImage.load(filename=WARP_CLEAN_FACE), FaceWarpedImage.load(filename=WARP_ONE_FACE)]
currDir = os.path.dirname(__file__)
pathToStoredIndex = os.path.join(currDir, "stored.index")


class TestIndexFunctionality(BaseTestClass):
    """Test of indexes."""

    descriptorVersion: int
    nonDefaultDescriptorVersion: int
    defaultFaceEstimator: FaceDescriptorEstimator
    faceDescriptor: FaceDescriptor
    nonDefaultFaceDescriptor: FaceDescriptor
    faceDescriptorBatch: FaceDescriptorBatch
    indexBuilder: IndexBuilder

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.descriptorVersion = EFDVa[0]
        cls.nonDefaultDescriptorVersion = EFDVa[1]
        cls.defaultFaceEstimator = BaseTestClass.faceEngine.createFaceDescriptorEstimator(cls.descriptorVersion)

        nonDefaultEstimator = BaseTestClass.faceEngine.createFaceDescriptorEstimator(EFDVa[1])
        cls.nonDefaultFaceDescriptor = nonDefaultEstimator.estimate(faceWarp)

    def setUp(self) -> None:
        super().setUp()
        self.indexBuilder = self.faceEngine.createIndexBuilder(self.descriptorVersion)
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
        assert isinstance(dynamicIndex, DynamicIndex), f"created {dynamicIndex.__class__} is not {DynamicIndex}"
        assert isinstance(dynamicIndex.bufSize, int), f"expected int but found {dynamicIndex.bufSize.__class__}"
        assert isinstance(dynamicIndex.descriptorsCount, int), \
            f"expected int but found {dynamicIndex.descriptorsCount.__class__}"
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
            # todo: remove after fix FSDK-2897
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
        nonExistentIndex = 2
        # todo: change test after FSDK-2897
        with self.assertRaises(IndexError) as ex:
            descriptor = self.indexBuilder[nonExistentIndex]
        assert str(nonExistentIndex) in ex.exception.args[0]

    def test_append_non_default_descriptor_to_builder(self):
        """Test append non default descriptor to internal storage."""
        with pytest.raises(LunaSDKException) as ex:
            self.indexBuilder.append(self.nonDefaultFaceDescriptor)
        self.assertLunaVlError(ex, LunaVLError.IncompatibleDescriptors.format(
            f"mismatch of descriptor versions: expected={self.descriptorVersion} "
            f"received={self.nonDefaultDescriptorVersion}"
        ))

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
        nonExistentIndex = 2
        # todo: change test after FSDK-2897
        with self.assertRaises(IndexError) as ex:
            del self.indexBuilder[nonExistentIndex]
        assert str(nonExistentIndex) in ex.exception.args[0]

    @pytest.mark.skip("FSDK-2877 Segmentation fault")
    def test_append_descriptor_to_empty_dynamic_index(self):
        """Test append descriptor to empty dynamic index."""
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=0, expectedBufSize=0)
        dynamicIndex.append(self.faceDescriptor)
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=1, expectedBufSize=1)

    def test_append_descriptor_to_dynamic_index(self):
        """Test append descriptor to dynamic index."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=1, expectedBufSize=1)
        dynamicIndex.append(self.faceDescriptor)
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=2, expectedBufSize=2)

    def test_append_descriptors_batch_to_dynamic_index(self):
        """Test append descriptor batch to dynamic index."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=1, expectedBufSize=1)
        dynamicIndex.appendBatch(self.faceDescriptorBatch)
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=3, expectedBufSize=3)

    def test_get_descriptor_from_dynamic_index(self):
        """Test get descriptor from dynamic index."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        dynamicIndex = self.indexBuilder.buildIndex()
        for idx in range(len(self.faceDescriptorBatch)):
            with self.subTest(case=f"get descriptor with index: {idx}"):
                descriptor = dynamicIndex[idx]
                assert self.faceDescriptorBatch[idx].rawDescriptor == descriptor.rawDescriptor

    def test_remove_descriptor_from_dynamic_index(self):
        """Test remove descriptor from dynamic index."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=2, expectedBufSize=2)
        for index, expectedDescriptorCount in ((0, 1), (0, 0)):
            with self.subTest(case=f"remove descriptor with index: {index}"):
                del dynamicIndex[index]
                self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount, len(self.faceDescriptorBatch))

    def test_search_similar_descriptor(self):
        """Test search result for descriptors."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=2, expectedBufSize=2)
        result = dynamicIndex.search(self.faceDescriptor, len(self.faceDescriptorBatch))
        assert 2 == len(result), result
        assert result[0].distance < result[1].distance, "first result should be the minimum distance"
        topResult = result[0]
        assert isinstance(topResult, IndexResult), f"result {topResult.__class__} is not {IndexResult}"
        assert isinstance(topResult.distance, float), f"expected float but found {topResult.distance.__class__}"
        assert isinstance(topResult.similarity, float), f"expected float but found {topResult.similarity.__class__}"
        assert isinstance(topResult.index, int), f"expected int but found {topResult.index.__class__}"
        assert len(self.faceDescriptorBatch) > topResult.index, "dynamic index out of range"
        assert 0.0 < topResult.similarity <= 1.0, "similarity out of range [0,1]"

    def test_search_result_non_default_descriptor(self):
        """Test search result by descriptor different version."""
        self.indexBuilder.appendBatch(self.faceDescriptorBatch)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=2, expectedBufSize=2)
        with pytest.raises(LunaSDKException) as ex:
            dynamicIndex.search(self.nonDefaultFaceDescriptor)
        self.assertLunaVlError(ex, LunaVLError.IncompatibleDescriptors.format(
            f"mismatch of descriptor versions: expected={self.descriptorVersion} "
            f"received={self.nonDefaultDescriptorVersion}"
        ))

    @pytest.mark.skip("FSDK-2897 internal error")
    def test_search_result_empty(self):
        """Test search with empty result."""
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=0, expectedBufSize=0)
        result = dynamicIndex.search(self.faceDescriptor)
        assert [] == result

    @pytest.mark.skip("FSDK-2897 failed search index")
    def test_search_result_invalid_input(self):
        """Test search with invalid parameter."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=1, expectedBufSize=1)
        with pytest.raises(LunaSDKException) as ex:
            dynamicIndex.search(self.faceDescriptor, maxCount=0)
        self.assertLunaVlError(ex, LunaVLError.InvalidInput.format("Invalid input"))

    def test_save_and_load_dynamic_index(self):
        """Test save and load dynamic index."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=1, expectedBufSize=1)

        dynamicIndex.save(pathToStoredIndex, IndexType.dynamic)
        assert os.path.isfile(pathToStoredIndex), "dynamic index file not found"
        dynamicIndex = self.indexBuilder.loadIndex(pathToStoredIndex, IndexType.dynamic)
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=1, expectedBufSize=1)

    def test_save_and_load_dense_index(self):
        """Test save and load dense index."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=1, expectedBufSize=1)

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
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=1, expectedBufSize=1)

        for path in (os.path.join(currDir, ".."), os.path.join(currDir, "./"), currDir):
            with self.subTest(path=path):
                with self.assertRaises(ValueError) as ex:
                    dynamicIndex.save(path, IndexType.dynamic)
                assert f"{path} must not be a directory" in ex.exception.args[0]

    def test_save_index_invalid_index_type(self):
        """Test save index with invalid index type."""
        self.indexBuilder.append(self.faceDescriptor)
        dynamicIndex = self.indexBuilder.buildIndex()
        self.assertDynamicIndex(dynamicIndex, expectedDescriptorCount=1, expectedBufSize=1)

        with pytest.raises(ValueError) as ex:
            dynamicIndex.save(pathToStoredIndex, "someType")
        assert ex.value.args[0] == "'someType' is not a valid IndexType"

    @pytest.mark.skip("FSDK-2897 RuntimeError: Failed to read index metadata")
    def test_load_index_unknown_file(self):
        """Test load index unknown file."""
        with pytest.raises(LunaSDKException) as ex:
            self.indexBuilder.loadIndex(WARP_ONE_FACE, IndexType.dynamic)
        self.assertLunaVlError(ex, LunaVLError.InvalidInput)
