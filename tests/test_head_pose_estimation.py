from collections import namedtuple
import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.base import ImageWithFaceDetection
from lunavl.sdk.estimators.face_estimators.head_pose import HeadPoseEstimator, HeadPose, FrontalType
from lunavl.sdk.detectors.facedetector import FaceDetector, FaceDetection
from lunavl.sdk.base import BoundingBox
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage

from FaceEngine import Detection, RectFloat

from tests.base import BaseTestClass
from tests.resources import (
    ONE_FACE,
    NO_FACES,
    GOST_HEAD_POSE_FACE,
    TURNED_HEAD_POSE_FACE,
    FRONTAL_HEAD_POSE_FACE,
)


class TestHeadPose(BaseTestClass):
    """
    Test of detector.
    """

    #: Face detector
    detector: FaceDetector
    #: head pose estimator
    headPoseEstimator: HeadPoseEstimator
    #: default image
    image: VLImage
    #: detection on default image
    detection: FaceDetection

    @classmethod
    def setup_class(cls):
        """
        Set up a data for tests.
        Create detection for estimations.
        """
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.headPoseEstimator = cls.faceEngine.createHeadPoseEstimator()
        cls.image = VLImage.load(filename=ONE_FACE)
        cls.detection = TestHeadPose.detector.detectOne(cls.image, detect5Landmarks=True, detect68Landmarks=True)

    def assertAngle(self, angle: float):
        """
        Assert degree measure of an angle

        Args:
            angle: -180 <= number <= 180
        """
        assert isinstance(angle, float), f"bad angle instance, type of angle is {type(angle)}"
        assert -180 <= angle <= 180

    def assertHeadPose(self, headPose: HeadPose):
        """
        Assert head pose estimation.

        Args:
            headPose: an estimate head pose
        """
        assert isinstance(headPose, HeadPose), f"bad head pose instance, type of head pose is {type(headPose)}"

        self.assertAngle(headPose.yaw)
        self.assertAngle(headPose.roll)
        self.assertAngle(headPose.pitch)

        assert isinstance(
            headPose.getFrontalType(), FrontalType
        ), f"bad instance frontal type, type of frontal type is {type(headPose.getFrontalType())}"

    def test_init_estimator(self):
        """
        Test init estimator.
        """
        estimator = TestHeadPose.faceEngine.createHeadPoseEstimator()
        assert isinstance(
            estimator, HeadPoseEstimator
        ), f"bad estimator instance, type of estimator  is {type(HeadPoseEstimator)}"

    def test_estimate_head_pose_by_68landmarks(self):
        """
        Estimating head pose by 68 landmarks test.
        """
        angles = TestHeadPose.headPoseEstimator.estimateBy68Landmarks(self.detection.landmarks68)
        self.assertHeadPose(angles)

    def test_estimate_head_pose_by_bounding_box(self):
        """
        Estimating head pose by bounding box test.
        """
        angles = TestHeadPose.headPoseEstimator.estimateByBoundingBox(
            ImageWithFaceDetection(self.image, self.detection.boundingBox)
        )
        self.assertHeadPose(angles)

    def test_estimate_head_pose_by_bounding_box_from_other_image(self):
        """
        Estimating head pose on image without faces by bounding box from other image.
        """
        image = VLImage.load(filename=NO_FACES)
        angles = TestHeadPose.headPoseEstimator.estimateByBoundingBox(
            ImageWithFaceDetection(self.image, self.detection.boundingBox)
        )
        self.assertHeadPose(angles)

    def test_estimate_head_pose_by_image_and_bounding_box_without_intersection(self):
        """
        Estimating head pose by image and bounding box without intersection
        """
        fakeDetection = Detection(RectFloat(3000.0, 3000.0, 100.0, 100.0), 0.9)
        bBox = BoundingBox(fakeDetection)
        with pytest.raises(LunaSDKException) as exceptionInfo:
            TestHeadPose.headPoseEstimator.estimateByBoundingBox(ImageWithFaceDetection(self.image, bBox))
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidRect.format("Invalid rectangle"))

    def test_estimate_head_pose_by_image_and_bounding_box_empty_bounding_box(self):
        """
        Estimating head pose by image and empty bounding box
        """
        fakeDetection = Detection(RectFloat(0.0, 0.0, 0.0, 0.0), 0.9)
        bBox = BoundingBox(fakeDetection)
        with pytest.raises(LunaSDKException) as exceptionInfo:
            TestHeadPose.headPoseEstimator.estimateByBoundingBox(ImageWithFaceDetection(self.image, bBox))
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidDetection.format("Invalid detection"))

    def test_default_estimation(self):
        """
        Default estimating head pose test.
        """
        angles1 = TestHeadPose.headPoseEstimator.estimateBy68Landmarks(self.detection.landmarks68)
        angles2 = TestHeadPose.headPoseEstimator.estimate(self.detection.landmarks68)
        assert angles1.pitch == angles2.pitch
        assert angles1.roll == angles2.roll
        assert angles1.yaw == angles2.yaw

    def test_head_pose_as_dict(self):
        """
        Test for a method asDict.
        """
        angles = TestHeadPose.headPoseEstimator.estimateBy68Landmarks(self.detection.landmarks68)
        self.assertHeadPose(angles)
        assert {"pitch": angles.pitch, "roll": angles.roll, "yaw": angles.yaw} == angles.asDict()

    def test_frontal_type(self):
        """
        Frontal types test.
        """
        Case = namedtuple("Case", ("image", "type"))
        cases = (
            Case(VLImage.load(filename=GOST_HEAD_POSE_FACE), FrontalType.BY_GOST),
            Case(VLImage.load(filename=TURNED_HEAD_POSE_FACE), FrontalType.TURNED),
            Case(VLImage.load(filename=FRONTAL_HEAD_POSE_FACE), FrontalType.FRONTAL),
        )
        for case in cases:
            with self.subTest(type=case.type):
                detection = TestHeadPose.detector.detectOne(case.image, detect5Landmarks=True, detect68Landmarks=True)
                angles = TestHeadPose.headPoseEstimator.estimateBy68Landmarks(detection.landmarks68)
                self.assertHeadPose(angles)
                assert angles.getFrontalType() == case.type

    def test_head_pose_batch_estimation(self):
        """
        Batch estimating head pose test.
        """
        image = VLImage.load(filename=GOST_HEAD_POSE_FACE)
        detection = self.detector.detectOne(image, detect5Landmarks=True, detect68Landmarks=True)
        imageWithFaceDetectionList = [
            ImageWithFaceDetection(self.image, self.detection.boundingBox),
            ImageWithFaceDetection(image, detection.boundingBox),
        ]
        anglesList = self.headPoseEstimator.estimateBatch(imageWithFaceDetectionList)
        assert isinstance(anglesList, list)
        assert len(anglesList) == 2
        for angles in anglesList:
            self.assertHeadPose(angles)

    def test_head_pose_batch_invalid_input(self):
        """
        Batch estimation invalid input
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.headPoseEstimator.estimateBatch([], [])
        self.assertLunaVlError(exceptionInfo, LunaVLError.InvalidSpanSize.format("Invalid span size"))

    def test_async_estimate_head_pose(self):
        """
        Test async estimate head pose
        """
        task = self.headPoseEstimator.estimate(self.detection.landmarks68, asyncEstimate=True)
        self.assertAsyncEstimation(task, HeadPose)
        task = self.headPoseEstimator.estimateBatch([self.detection] * 2, asyncEstimate=True)
        self.assertAsyncBatchEstimation(task, HeadPose)
        task = self.headPoseEstimator.estimateByBoundingBox(
            ImageWithFaceDetection(self.image, self.detection.boundingBox), asyncEstimate=True
        )
        self.assertAsyncEstimation(task, HeadPose)
        task = self.headPoseEstimator.estimateBy68Landmarks(self.detection.landmarks68, asyncEstimate=True)
        self.assertAsyncEstimation(task, HeadPose)
