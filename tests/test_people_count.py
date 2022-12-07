import pytest
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from tests.base import BaseTestClass
from lunavl.sdk.estimators.image_estimators.people_count import ImageForPeopleEstimation
from lunavl.sdk.faceengine.setting_provider import PeopleCountEstimatorType
from lunavl.sdk.image_utils.image import VLImage, ColorFormat
from lunavl.sdk.image_utils.geometry import Rect
from tests.resources import CROWD_9_PEOPLE, CROWD_7_PEOPLE, IMAGE_WITH_TWO_FACES

class TestPeopleCount(BaseTestClass):
    """
    Test estimate people count
    """
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.peopleCountEstimator = cls.faceEngine.createPeopleCountEstimator(PeopleCountEstimatorType.CROWD_AND_HEAD)
        cls.crowd9People = VLImage.load(filename=CROWD_9_PEOPLE)
        cls.crowd7People = VLImage.load(filename=CROWD_7_PEOPLE)
        cls.badFormatImage = VLImage.load(filename=CROWD_7_PEOPLE, colorFormat=ColorFormat.B8G8R8)
        cls.outsideArea = ImageForPeopleEstimation(
            cls.crowd9People,
            Rect(100, 100, cls.crowd9People.rect.width, cls.crowd9People.rect.height)
        )
        cls.areaLargerImage = ImageForPeopleEstimation(
            cls.crowd9People,
            Rect(100, 100, cls.crowd9People.rect.width + 100, cls.crowd9People.rect.height + 100)
        )
        cls.areaOutsideImage = ImageForPeopleEstimation(
            cls.crowd9People,
            Rect(
                cls.crowd9People.rect.width,
                cls.crowd9People.rect.height,
                cls.crowd9People.rect.width + 100,
                cls.crowd9People.rect.height + 100,
            )
        )
        cls.areaWithoutPeople = ImageForPeopleEstimation(
            cls.crowd9People,
            Rect(10, 10, 100, 100)
        )
        cls.invalidRectImage = ImageForPeopleEstimation(
            cls.crowd9People,
            Rect(0, 0, 0, 0)
        )
        cls.errorCoreRectImage = ImageForPeopleEstimation(
            cls.crowd9People,
            Rect(0.1, 0.1, 0.1, 0.1)
        )

    def test_people_count_async(self):
        """
        Test single image async estimation
        """
        peopleCount = self.peopleCountEstimator.estimate(self.crowd9People, asyncEstimate=True).get()
        assert peopleCount == 9

    def test_people_count(self):
        """
        Test single image estimation
        """
        peopleCount = self.peopleCountEstimator.estimate(self.crowd7People)
        assert peopleCount == 7

    def test_people_count_batch(self):
        """
        Test batch estimation
        """
        images = [
            self.crowd9People,
            VLImage.load(filename=IMAGE_WITH_TWO_FACES),
            self.crowd7People
        ]
        peopleCount = self.peopleCountEstimator.estimateBatch(images)
        assert peopleCount == [9, 2, 7]

    def test_people_count_with_batch_invalid_input(self):
        """
        Test estimation with bad input
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimateBatch([])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidSpanSize.format("Invalid span size"))

    def test_people_count_with_bad_format_image(self):
        """
        Test estimation with unsupported image format
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(self.badFormatImage)
        detail = f"Bad image format for people estimation," \
                 f" format: {self.badFormatImage.format.value}," \
                 f" image: {self.badFormatImage.filename}"
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidImageFormat.format(detail))

    def test_people_count_batch_with_bad_format_image(self):
        """
        Test batch estimation with unsupported image format
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimateBatch([self.crowd9People, self.badFormatImage])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[1], LunaVLError.InvalidImageFormat)

    def test_people_count_with_area_outside(self):
        """
        Test estimation with area slightly outside image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(self.outsideArea)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.InvalidRect)

    def test_people_count_batch_with_area_outside(self):
        """
        Test batch estimation with area slightly outside image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimateBatch([self.crowd7People, self.outsideArea])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[1], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_with_area_larger_image(self):
        """
        Test estimation with area is larger than image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(self.areaLargerImage)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[0], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_batch_with_area_larger_image(self):
        """
        Test batch estimation with area is larger than image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimateBatch([self.crowd7People, self.areaLargerImage])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[1], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_with_area_outside_image(self):
        """
        Test estimation with area completely outside image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(self.areaOutsideImage)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[0], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_batch_with_area_outside_image(self):
        """
        Test batch estimation with area completely outside image
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimateBatch([self.crowd7People, self.areaOutsideImage])
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[1], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_batch_with_area_without_people(self):
        """
        Test estimation with not contain people area
        """
        peopleCount = self.peopleCountEstimator.estimateBatch([self.areaWithoutPeople, self.crowd7People])
        assert peopleCount == [0, 7]

    def test_people_count_with_invalid_area(self):
        """
        Test estimation with invalid rectangle
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(self.invalidRectImage)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[0], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

    def test_people_count_with_invalid_core_rect(self):
        """
        Test estimation with invalid core rectangle
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimate(self.errorCoreRectImage)
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        self.assertReceivedAndRawExpectedErrors(
            exceptionInfo.value.context[0], LunaVLError.InvalidRect.format("Invalid rectangle")
        )

