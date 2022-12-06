import pytest
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from tests.base import BaseTestClass
from lunavl.sdk.faceengine.setting_provider import PeopleCountEstimatorType
from lunavl.sdk.image_utils.image import VLImage
from tests.resources import CROWD_9_PEOPLE, CROWD_7_PEOPLE, IMAGE_WITH_TWO_FACES

class TestPeopleCount(BaseTestClass):
    """
    Test estimate people count
    """
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.peopleCountEstimator = cls.faceEngine.createPeopleCountEstimator(PeopleCountEstimatorType.CROWD_AND_HEAD)

    def test_people_count_async(self):
        """
        Test single image async estimation
        """
        image = VLImage.load(filename=CROWD_9_PEOPLE)
        peopleCount = self.peopleCountEstimator.estimate(image, asyncEstimate=True).get()
        assert peopleCount == 9

    def test_people_count(self):
        """
        Test single image estimation
        """
        image = VLImage.load(filename=CROWD_7_PEOPLE)
        peopleCount = self.peopleCountEstimator.estimate(image)
        assert peopleCount == 7

    def test_people_count_batch(self):
        """
        Test batch estimation
        """
        images = [
            VLImage.load(filename=CROWD_9_PEOPLE),
            VLImage.load(filename=IMAGE_WITH_TWO_FACES),
            VLImage.load(filename=CROWD_7_PEOPLE),
        ]
        peopleCount = self.peopleCountEstimator.estimateBatch(images)
        assert peopleCount == [9, 2, 7]

    def test_people_count_with_batch_invalid_input(self):
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.peopleCountEstimator.estimateBatch([])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidSpanSize.format("Invalid span size"))