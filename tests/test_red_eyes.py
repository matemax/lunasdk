import pytest

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.eyes import WarpWithLandmarks5
from lunavl.sdk.estimators.face_estimators.red_eye import RedEyes
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import MANY_FACES, OPEN_EYES, RED_EYES

OPEN_EYES_IMAGE = VLImage.load(filename=OPEN_EYES)
RED_EYES_IMAGE = VLImage.load(filename=RED_EYES)


class TestEstimateRedEyes(BaseTestClass):
    """
    Test estimate red-eyes.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.redEyeEstimator = cls.faceEngine.createRedEyeEstimator()

    def estimate(self, image: VLImage):
        """Estimate red eyes om image"""
        faceDetection = self.detector.detectOne(image)
        warp = self.warper.warp(faceDetection)
        landMarks5Transformation = self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")
        warpWithLandmarks = WarpWithLandmarks5(warp, landMarks5Transformation)
        redEyes = self.redEyeEstimator.estimate(warpWithLandmarks)
        return redEyes

    def assertEstimation(self, estimation: RedEyes):
        """Assert red-eyes estimation"""
        assert isinstance(estimation, RedEyes)
        for eye in (estimation.leftEye, estimation.rightEye):
            assert 0 <= eye.score <= 1
            assert isinstance(eye.status, bool)

    def test_estimate_red_eyes(self):
        """
        Test estimate red eyes
        """
        redEyes = self.estimate(RED_EYES_IMAGE)
        self.assertEstimation(redEyes)
        assert redEyes.rightEye.status
        assert redEyes.leftEye.status

    def test_estimate_open_eyes(self):
        """
        Test eye estimator with face with opened eyes
        """
        redEyes = self.estimate(OPEN_EYES_IMAGE)
        self.assertEstimation(redEyes)
        assert not redEyes.rightEye.status
        assert not redEyes.leftEye.status

    def test_red_eyes_as_dict(self):
        """
        Test eye estimator with face with closed eyes
        """
        redEyes = self.estimate(OPEN_EYES_IMAGE)
        self.assertEstimation(redEyes)
        assert {
            "left_eye": {
                "status": redEyes.leftEye.status,
                "score": redEyes.leftEye.score,
            },
            "right_eye": {
                "status": redEyes.rightEye.status,
                "score": redEyes.rightEye.score,
            },
        } == redEyes.asDict()

    def test_estimate_batch(self):
        """
        Test eye estimator with two faces
        """
        faceDetections = self.detector.detect([RED_EYES_IMAGE, OPEN_EYES_IMAGE], detect5Landmarks=True)
        faceDetections = [faceDetections[0][0], faceDetections[1][0]]
        warpWithLandmarksList = [
            WarpWithLandmarks5(
                self.warper.warp(faceDetection),
                self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L5"),
            )
            for faceDetection in faceDetections
        ]
        estimations = self.redEyeEstimator.estimateBatch(warpWithLandmarksList)
        assert isinstance(estimations, list)
        assert len(estimations) == len(faceDetections)
        for estimation in estimations:
            self.assertEstimation(estimation)
        assert estimations[0].rightEye.status
        assert not estimations[1].rightEye.status

    def test_estimate_batch_invalid_input(self):
        """
        Test batch eye estimator with invalid input
        """
        with pytest.raises(LunaSDKException) as e:
            self.redEyeEstimator.estimateBatch([], [])
        assert e.value.error.errorCode == LunaVLError.InvalidSpanSize.errorCode

    def test_bad_landmarks5(self):
        """Try estimate red eyes with incorrect landmarks"""
        faceDetection = self.detector.detectOne(VLImage.load(filename=MANY_FACES))
        warp = self.warper.warp(faceDetection)
        warpWithLandmarks = WarpWithLandmarks5(warp, faceDetection.landmarks5)
        redEyes = self.redEyeEstimator.estimate(warpWithLandmarks)
        assert isinstance(redEyes, RedEyes)

    def test_estimate_redeyes_batch(self):
        """
        Batch redeyes estimation test
        """

        faceDetections = self.detector.detect([OPEN_EYES_IMAGE, RED_EYES_IMAGE])

        warp1 = self.warper.warp(faceDetections[0][0])
        warp2 = self.warper.warp(faceDetections[1][0])
        landMarks5Transformation1 = self.warper.makeWarpTransformationWithLandmarks(faceDetections[0][0], "L5")
        landMarks5Transformation2 = self.warper.makeWarpTransformationWithLandmarks(faceDetections[1][0], "L5")
        warpWithLandmarks1 = WarpWithLandmarks5(warp1, landMarks5Transformation1)
        warpWithLandmarks2 = WarpWithLandmarks5(warp2, landMarks5Transformation2)
        estimations = self.redEyeEstimator.estimateBatch([warpWithLandmarks1, warpWithLandmarks2])
        assert not estimations[0].leftEye.status
        assert estimations[1].leftEye.status
