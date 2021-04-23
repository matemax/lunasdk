import pytest

from lunavl.sdk.estimators.face_estimators.eyes import EyeState, EyesEstimation, WarpWithLandmarks
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import OPEN_EYES, CLOSED_EYES, MIXED_EYES

OPEN_EYES_IMAGE = VLImage.load(filename=OPEN_EYES)
MIXED_EYES_IMAGE = VLImage.load(filename=MIXED_EYES)
CLOSED_EYES_IMAGE = VLImage.load(filename=CLOSED_EYES)


class TestEstimateEyes(BaseTestClass):
    """
    Test estimate eyes.
    """

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_DEFAULT)
        cls.warper = cls.faceEngine.createFaceWarper()
        cls.eyeEstimator = cls.faceEngine.createEyeEstimator()

    @staticmethod
    def validate_eye_dict(receivedDict: dict):
        """
        Validate emotion dict
        """

        def assert_landmarks(landmarks: dict) -> None:
            """
            Assert landmarks list of tuples
            """
            for _landmarks in landmarks:
                assert isinstance(_landmarks, tuple)
                assert len(_landmarks) == 2
                for landMark in _landmarks:
                    assert isinstance(landMark, int)

        assert {"iris_landmarks", "eyelid_landmarks", "state"} == receivedDict.keys()
        assert receivedDict["state"] in ("open", "occluded", "closed")
        assert len(receivedDict["iris_landmarks"]) == 32
        assert_landmarks(receivedDict["iris_landmarks"])
        assert len(receivedDict["eyelid_landmarks"]) == 6
        assert_landmarks(receivedDict["eyelid_landmarks"])

    def assert_eyes_reply(self, eyesDict: dict) -> None:
        """
        Assert eyes dict
        Args:
            eyesDict: dict with eyes detection reply
        """
        assert eyesDict.keys() == {"left_eye", "right_eye"}
        for eyeDict in eyesDict.values():
            self.validate_eye_dict(eyeDict)

    def test_estimate_eye_as_dict(self):
        """
        Test eye estimator 'asDict' method
        """
        faceDetection = self.detector.detectOne(OPEN_EYES_IMAGE)
        warp = self.warper.warp(faceDetection)
        landMarks5Transformation = self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")
        warpWithLandmarks = WarpWithLandmarks(warp.warpedImage.coreImage, landMarks5Transformation)
        eyesDict = self.eyeEstimator.estimate(warpWithLandmarks).asDict()
        self.assert_eyes_reply(eyesDict)

    def test_estimate_open_eyes(self):
        """
        Test eye estimator with face with opened eyes
        """
        faceDetection = self.detector.detectOne(OPEN_EYES_IMAGE)
        warp = self.warper.warp(faceDetection)
        landMarks5Transformation = self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")
        warpWithLandmarks = WarpWithLandmarks(warp.warpedImage.coreImage, landMarks5Transformation)
        eyesResult = self.eyeEstimator.estimate(warpWithLandmarks)
        assert eyesResult.leftEye.state == EyeState.Open
        assert eyesResult.rightEye.state == EyeState.Open

    def test_estimate_closed_eyes(self):
        """
        Test eye estimator with face with closed eyes
        """
        faceDetection = self.detector.detectOne(CLOSED_EYES_IMAGE)
        warp = self.warper.warp(faceDetection)
        landMarks5Transformation = self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")
        warpWithLandmarks = WarpWithLandmarks(warp.warpedImage.coreImage, landMarks5Transformation)
        eyesResult = self.eyeEstimator.estimate(warpWithLandmarks)
        assert eyesResult.leftEye.state == EyeState.Closed
        assert eyesResult.rightEye.state == EyeState.Closed

    def test_estimate_mixed_eyes(self):
        """
        Test eye estimator with face with mixed eyes
        """
        faceDetection = self.detector.detectOne(MIXED_EYES_IMAGE)
        warp = self.warper.warp(faceDetection)
        landMarks5Transformation = self.warper.makeWarpTransformationWithLandmarks(faceDetection, "L5")
        warpWithLandmarks = WarpWithLandmarks(warp.warpedImage.coreImage, landMarks5Transformation)
        eyesResult = self.eyeEstimator.estimate(warpWithLandmarks)
        assert eyesResult.leftEye.state == EyeState.Occluded
        assert eyesResult.rightEye.state == EyeState.Open

    def test_estimate_batch(self):
        """
        Test eye estimator with two faces
        """
        for landMarks in ("L5", "L68"):
            with self.subTest(landMarks=landMarks):
                faceDetections = [
                    self.detector.detectOne(img, detect68Landmarks=landMarks == "L68")
                    for img in (OPEN_EYES_IMAGE, MIXED_EYES_IMAGE, CLOSED_EYES_IMAGE)
                ]
                warpWithLandmarksList = [
                    WarpWithLandmarks(
                        self.warper.warp(faceDetection).warpedImage.coreImage,
                        self.warper.makeWarpTransformationWithLandmarks(faceDetection, landMarks),
                    )
                    for faceDetection in faceDetections
                ]
                anglesList = self.eyeEstimator.estimateBatch(warpWithLandmarksList)
                assert isinstance(anglesList, list)
                assert len(anglesList) == len(faceDetections)
                for angles in anglesList:
                    assert isinstance(angles, EyesEstimation)
                    self.assert_eyes_reply(angles.asDict())

    def test_estimate_batch_mixed_landmarks(self):
        """
        Test eye estimator with two faces - 1st with 5 landmarks, 2nd with 68 landmarks
        """
        faceDetections = [
            self.detector.detectOne(OPEN_EYES_IMAGE, detect68Landmarks=False),
            self.detector.detectOne(OPEN_EYES_IMAGE, detect68Landmarks=True),
        ]

        warpWithLandmarksList = [
            WarpWithLandmarks(
                self.warper.warp(faceDetections[0]).warpedImage.coreImage,
                self.warper.makeWarpTransformationWithLandmarks(faceDetections[0], "L5"),
            ),
            WarpWithLandmarks(
                self.warper.warp(faceDetections[1]).warpedImage.coreImage,
                self.warper.makeWarpTransformationWithLandmarks(faceDetections[1], "L68"),
            ),
        ]
        anglesList = self.eyeEstimator.estimateBatch(warpWithLandmarksList)
        assert isinstance(anglesList, list)
        assert len(anglesList) == len(faceDetections)
        for angles in anglesList:
            assert isinstance(angles, EyesEstimation)
            self.assert_eyes_reply(angles.asDict())

    def test_estimate_batch_invalid_input(self):
        """
        Test batch eye estimator with invalid input
        """
        with pytest.raises(TypeError):
            self.eyeEstimator.estimateBatch([], [])
