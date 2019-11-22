from typing import Union, List

import pytest

from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.basic_attributes import BasicAttributesEstimator, BasicAttributes
from lunavl.sdk.estimators.face_estimators.warper import Warper, Warp
from lunavl.sdk.faceengine.facedetector import (
    FaceDetector)
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import ONE_FACE


class TestEstimate(BaseTestClass):
    """
    Test of estimation.
    """

    detector: FaceDetector = None
    warper: Warper = None
    estimator: BasicAttributesEstimator = None

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.detector = cls.faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
        cls.warper = cls.faceEngine.createWarper()
        cls.estimator = cls.faceEngine.createBasicAttributesEstimator()

    @staticmethod
    def getBasicAttributeError(warp: Union[Warp, List], estimateAge: bool, estimateGender: bool, estimateEthnicity: bool):
        with pytest.raises(LunaSDKException) as ex:
            if isinstance(warp, list):
                TestEstimate.estimator.estimateBasicAttributesBatch(warps=warp,
                                                                    estimateAge=estimateAge,
                                                                    estimateEthnicity=estimateEthnicity,
                                                                    estimateGender=estimateGender)
            else:
                TestEstimate.estimator.estimate(warp=warp, estimateAge=estimateAge, estimateEthnicity=estimateEthnicity,
                                                estimateGender=estimateGender)
        error = {"error_code": ex.value.error.errorCode, "desc": ex.value.error.description,
                 "detail": ex.value.error.detail}
        return error

    def test_estimate_basic_attributes(self):
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        detect = TestEstimate.detector.detectOne(image=imageWithOneFace)
        warp = TestEstimate.warper.warp(detect)
        assert warp.warpedImage.assertWarp() is None
        estimateBasic = TestEstimate.estimator.estimate(warp=warp.warpedImage, estimateAge=True,
                                                        estimateEthnicity=True,
                                                        estimateGender=True)
        assert isinstance(estimateBasic, BasicAttributes)
        assert estimateBasic.asDict()['ethnicities'] is not None
        assert estimateBasic.asDict()['age'] is not None
        assert estimateBasic.asDict()['gender'] in (0, 1)

    def test_estimation_basic_bad_optional_attr(self):
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        detect = TestEstimate.detector.detectOne(image=imageWithOneFace)
        warp = TestEstimate.warper.warp(detect)
        assert warp.warpedImage.assertWarp() is None
        error = self.getBasicAttributeError(warp=warp, estimateAge=False, estimateEthnicity=True, estimateGender=True)
        assert error["error_code"] == 110006
        assert error["desc"] == "Estimation basic attributes error"
        assert error["detail"] == "bad_optional_access"

    def test_estimation_basic_attr_batch(self):
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        detect = TestEstimate.detector.detectOne(image=imageWithOneFace)
        warp = TestEstimate.warper.warp(detect)
        assert warp.warpedImage.assertWarp() is None
        aggregate = False
        for _ in range(2):
            estimateBasicBatch = TestEstimate.estimator.estimateBasicAttributesBatch(warps=[warp, warp.warpedImage],
                                                                                     estimateAge=True,
                                                                                     estimateEthnicity=True,
                                                                                     estimateGender=True,
                                                                                     aggregate=aggregate)
            assert all(isinstance(basic, BasicAttributes) for basic in estimateBasicBatch[0])
            assert 2 == len(estimateBasicBatch)
            if aggregate:
                assert isinstance(estimateBasicBatch[1], BasicAttributes)
            else:
                assert estimateBasicBatch[1] is None
            aggregate = True

    def test_estimation(self):
        imageWithOneFace = VLImage.load(filename=ONE_FACE)
        detect = TestEstimate.detector.detectOne(image=imageWithOneFace)
        warp = TestEstimate.warper.warp(detect)
        assert warp.warpedImage.assertWarp() is None
        error = self.getBasicAttributeError(warp=[warp, warp.warpedImage], estimateAge=False, estimateEthnicity=True,
                                            estimateGender=True)
        assert error["error_code"] == 110007, error
        assert error["desc"] == "Batch estimation basic attributes error"
        assert error["detail"] == "bad_optional_access"
