import pytest

from lunavl.sdk.detectors.facedetector import FaceDetection, FaceDetector, Landmarks5, Landmarks68, FaceLandmarks
from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.landmarks import FaceLandmarksEstimator, _prepareBatch
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ONE_FACE, SEVERAL_FACES, IMAGE_WITH_TWO_FACES


class TestFaceLandmarks(BaseTestClass):
    """
    Test face landmarks estimations.
    """

    #: Face detector
    detector: FaceDetector
    #: head pose estimator
    estimator: FaceLandmarksEstimator
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
        cls.estimator = cls.faceEngine.createFaceLandmarksEstimator()
        cls.image = VLImage.load(filename=ONE_FACE)
        cls.detection = cls.detector.detectOne(cls.image)

    def assertLandmarks5(self, landmarks: Landmarks5):
        """
        Assert landmarks5 estimation.

        Args:
            landmarks: an estimate landmarks5
        """
        assert isinstance(landmarks, Landmarks5), f"bad landmarks instance, type of landmarks is {type(landmarks)}"

        assert 5 == len(landmarks.points)

    def assertLandmarks68(self, landmarks: Landmarks68):
        """
        Assert landmarks5 estimation.

        Args:
            landmarks: an estimate landmarks5
        """
        assert isinstance(landmarks, Landmarks68), f"bad landmarks instance, type of landmarks is {type(Landmarks68)}"

        assert 68 == len(landmarks.points)

    def test_init_estimator(self):
        """
        Test init estimator.
        """
        estimator = self.faceEngine.createFaceLandmarksEstimator()
        assert isinstance(
            estimator, FaceLandmarksEstimator
        ), f"bad estimator instance, type of estimator  is {type(estimator)}"

    def test_estimate_landmarks5(self):
        """
        Estimating landmarks5.
        """
        detection = self.detector.detectOne(self.image, detect5Landmarks=True)
        landmarks = self.estimator.estimate(self.detection, landmarksType=FaceLandmarks.Landmarks5)
        self.assertLandmarks5(landmarks)
        assert detection.landmarks5.asDict() == landmarks.asDict()

    def test_estimate_landmarks68(self):
        """
        Estimating landmarks68.
        """
        detection = self.detector.detectOne(self.image, detect68Landmarks=True)
        landmarks = self.estimator.estimate(self.detection, landmarksType=FaceLandmarks.Landmarks68)
        self.assertLandmarks68(landmarks)
        assert detection.landmarks68.asDict() == landmarks.asDict()

    def test_estimate_landmarks5_batch(self):
        """
        Estimating landmarks5.
        """
        detection = self.detector.detectOne(self.image, detect5Landmarks=True)
        landmarks = self.estimator.estimate(self.detection, landmarksType=FaceLandmarks.Landmarks5)
        self.assertLandmarks5(landmarks)
        assert detection.landmarks5.asDict() == landmarks.asDict()

    def test_estimate_landmarks_batch(self):
        """
        Estimating landmarks68. Test correctness and order returning values.
        """
        image1 = VLImage.load(filename=SEVERAL_FACES)
        image2 = self.image
        image3 = VLImage.load(filename=IMAGE_WITH_TWO_FACES)
        for landmarksType in (FaceLandmarks.Landmarks5, FaceLandmarks.Landmarks68):
            if landmarksType == FaceLandmarks.Landmarks5:
                res = self.detector.detect([image1, image2, image3], detect5Landmarks=True)
            else:
                res = self.detector.detect([image1, image2, image3], detect68Landmarks=True)
            faceDetections = []
            for imageDetections in res:
                faceDetections.extend(imageDetections)

            testCases = faceDetections.copy(), faceDetections.copy()[::2] + faceDetections.copy()[1::2]
            for idx, testCase in enumerate(testCases):
                with self.subTest(lanmarksType=landmarksType, testCaseIdx=idx):
                    estimations = self.estimator.estimateBatch(testCase, landmarksType=landmarksType)
                    for idx, landmarks in enumerate(estimations):
                        if landmarksType == FaceLandmarks.Landmarks5:
                            self.assertLandmarks5(landmarks)
                            assert testCase[idx].landmarks5.asDict() == landmarks.asDict()
                        else:
                            self.assertLandmarks68(landmarks)
                            assert testCase[idx].landmarks68.asDict() == landmarks.asDict()

    def test_batch_invalid_input(self):
        """
        Batch estimation invalid input
        """
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.estimator.estimateBatch([], [])
        self.assertLunaVlError(exceptionInfo, LunaVLError.ValidationFailed.format("Invalid span size"))

    def test_async_estimate_head_pose(self):
        """
        Test async estimate head pose
        """
        for landmarksType in (FaceLandmarks.Landmarks5, FaceLandmarks.Landmarks68):
            with self.subTest(lanmarksType=landmarksType):
                cls = Landmarks5 if landmarksType == FaceLandmarks.Landmarks5 else Landmarks68
                task = self.estimator.estimate(self.detection, landmarksType=landmarksType, asyncEstimate=True)
                self.assertAsyncEstimation(task, cls)
                task = self.estimator.estimateBatch(
                    [self.detection] * 2, landmarksType=FaceLandmarks.Landmarks5, asyncEstimate=True
                )
                self.assertAsyncBatchEstimation(task, cls)

    def test_prepare_batch_image_aggregation(self):
        """
        Unittest for `_prepareBatch` function.  Check correctness detection aggregation by image.
        """
        image1 = VLImage.load(filename=SEVERAL_FACES)
        image2 = self.image
        image3 = VLImage.load(filename=IMAGE_WITH_TWO_FACES)
        res = self.detector.detect([image1, image2, image3], detect5Landmarks=True)

        detections = {id(image1): [], id(image2): [], id(image3): []}
        allDetections = []
        for imageDetections in res:
            for detection in imageDetections:
                detections[id(imageDetections[0].image)].append(detection)
                allDetections.append(detection)
        testCases = allDetections.copy(), allDetections.copy()[::2] + allDetections.copy()[1::2]
        for index, testCase in enumerate(testCases):
            with self.subTest(index):
                preparedBatch = _prepareBatch(testCase)
                assert 3 == len(preparedBatch)
                for coreImage, coreDetections, originalIdx in preparedBatch:
                    for image in (image1, image2, image3):
                        if image.coreImage == coreImage:
                            break
                    else:
                        self.fail("original image not found")
                    assert len(detections[id(image)]) == len(coreDetections)
                    assert set([id(detection.coreEstimation.detection) for detection in detections[id(image)]]) == set(
                        [id(detection) for detection in coreDetections]
                    )
