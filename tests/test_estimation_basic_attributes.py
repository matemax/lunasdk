from collections import namedtuple
from dataclasses import dataclass
from operator import attrgetter
from time import time
from typing import List, Optional, Union, Callable, Tuple

import FaceEngine

from lunavl.sdk.estimators.face_estimators.basic_attributes import (
    BasicAttributesEstimator,
    BasicAttributes,
    Ethnicities,
)
from lunavl.sdk.estimators.face_estimators.warper import Warp
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.resources import MANY_FACES


def generateWarps(faceEngine: FaceEngine, imagePath: Optional[str] = MANY_FACES) -> List[Warp]:
    """
    Generate warps.

    Args:
        faceEngine: Face Engine instance
        imagePath: path to image

    Returns:
        first 1000 warps sorted by coordinates
    """
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    warper = faceEngine.createWarper()

    img = VLImage.load(filename=imagePath)
    detections = detector.detect(images=[img], limit=1000, detect68Landmarks=True)[0]

    def sortKey(warp: Warp):
        landmarks = warp.sourceDetection.landmarks68
        if landmarks is not None:
            point = landmarks.points[0]
            return point.x, point.y

    warps = sorted([warper.warp(detection) for detection in detections], key=sortKey)
    return warps


@dataclass
class Eth:
    """ Class for Ethnicities estimation. """

    asian: float
    indian: float
    caucasian: float
    africanAmerican: float

    def __eq__(self, other):
        return all(getattr(self, name) == getattr(other, name) for name in self.__dict__)

    def __sub__(self, other):
        return sum((getattr(self, name) - getattr(other, name)) ** 2 for name in self.__dict__) ** (1 / 2) / len(
            self.__dict__
        )


Estimation = namedtuple("Estimation", ("Age", "Gender", "Ethnicity"))


class TestBasicAttributes(BaseTestClass):
    """ Test basic attributes. """

    estimator: BasicAttributesEstimator = BaseTestClass.faceEngine.createBasicAttributesEstimator()

    _warps: List[Warp]

    @classmethod
    def setUpClass(cls) -> None:
        """ Load warps. """
        cls._warps = generateWarps(cls.faceEngine)

    def estimate(
        self, warp: Warp, estimateAge: bool = False, estimateGender: bool = False, estimateEthnicity: bool = False
    ) -> BasicAttributes:
        """
        Estimate warp.

        Args:
            warp: warp
            estimateAge: whether to estimate age
            estimateGender: whether to estimate gender
            estimateEthnicity: whether to estimate ethnicity

        Returns:
             estimated attributes
        """
        return self.estimator.estimate(
            warp=warp, estimateAge=estimateAge, estimateGender=estimateGender, estimateEthnicity=estimateEthnicity
        )

    def estimateBatch(
        self,
        warps: List[Warp],
        estimateAge: bool = False,
        estimateGender: bool = False,
        estimateEthnicity: bool = False,
        aggregate: bool = False,
    ) -> Tuple[List[BasicAttributes], Union[None, BasicAttributes]]:
        """
        Estimate batch of warps.

        Args:
            warps: warps
            estimateAge: whether to estimate age
            estimateGender: whether to estimate gender
            estimateEthnicity: whether to estimate ethnicity
            aggregate: whether to aggregate

        Returns:
            tuple: list of estimated attributes and aggregated attributes
        """
        return self.estimator.estimateBasicAttributesBatch(
            warps=warps,
            estimateAge=estimateAge,
            estimateGender=estimateGender,
            estimateEthnicity=estimateEthnicity,
            aggregate=aggregate,
        )

    def test_correctness(self):
        """
        Test estimation correctness.
        """
        AGE_DELTA = 3
        GENDER_DELTA = 0
        ETH_DELTA = 0.001
        warpNumberToEstimations = {
            0: Estimation(37, 1, Eth(0, 0, 1, 0)),
            1: Estimation(23, 1, Eth(0, 0, 1, 0)),
            2: Estimation(29, 0, Eth(0, 0, 1, 0)),
            3: Estimation(43, 1, Eth(0, 0, 1, 0)),
            4: Estimation(50, 1, Eth(0.0005, 0.0076, 0.247, 0.7445)),
            5: Estimation(25, 1, Eth(0, 0, 1, 0)),
            6: Estimation(47, 1, Eth(0, 0, 0, 1)),
            7: Estimation(35, 1, Eth(0, 0, 1, 0)),
            8: Estimation(34, 0, Eth(0, 0, 1, 0)),
            9: Estimation(47, 1, Eth(0, 0.004, 0.995, 0)),
            10: Estimation(34, 1, Eth(0, 0, 1, 0)),
            11: Estimation(34, 1, Eth(0, 0, 1, 0)),
            12: Estimation(54, 1, Eth(0, 0, 1, 0)),
            13: Estimation(33, 1, Eth(0, 0, 0, 1)),
            14: Estimation(34, 1, Eth(0, 0, 1, 0)),
            15: Estimation(36, 1, Eth(0, 0, 0, 1)),
            16: Estimation(26, 0, Eth(0, 0, 1, 0)),
            17: Estimation(29, 1, Eth(0, 0, 1, 0)),
            18: Estimation(30, 0, Eth(1, 0, 0, 0)),
            19: Estimation(42, 1, Eth(0, 0, 1, 0)),
        }
        assert len(self._warps) == len(warpNumberToEstimations)
        for estimationType, delta in (("Age", AGE_DELTA), ("Gender", GENDER_DELTA), ("Ethnicity", ETH_DELTA)):
            estimationFlag = f"estimate{estimationType}"
            basicAttributeGetter: Callable[[BasicAttributes], Union[Ethnicities, float, None]] = attrgetter(
                estimationType.lower()
            )
            for warpNumber, warp in enumerate(self._warps):
                expectedAttr = getattr(warpNumberToEstimations[warpNumber], estimationType)
                with self.subTest(estimationType=estimationType, warpNumber=warpNumber):
                    singleAttr = basicAttributeGetter(self.estimate(warp, **{estimationFlag: True}))
                    batchAttr = basicAttributeGetter(self.estimateBatch([warp], **{estimationFlag: True})[0][0])
                    self.assertNotIn(None, (singleAttr, batchAttr))

                    filename = f"./{time()}.jpg"
                    msg = f"Batch estimation '{estimationType}' differs from single one. Saved as '{filename}'."
                    try:
                        if isinstance(singleAttr, Ethnicities):
                            self.assertEqual(singleAttr.asian, batchAttr.asian, msg)
                            self.assertEqual(singleAttr.indian, batchAttr.indian, msg)
                            self.assertEqual(singleAttr.caucasian, batchAttr.caucasian, msg)
                            self.assertEqual(singleAttr.africanAmerican, batchAttr.africanAmerican, msg)
                        else:
                            self.assertEqual(singleAttr, batchAttr, msg)
                    except AssertionError:
                        warp.warpedImage.save(filename)
                        raise

                    self.assertAlmostEqual(expectedAttr, singleAttr, delta=delta)
