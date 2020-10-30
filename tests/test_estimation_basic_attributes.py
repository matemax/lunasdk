from collections import namedtuple
from dataclasses import dataclass
from operator import attrgetter
from time import time
from typing import List, Optional, Union, Callable, Tuple

from lunavl.sdk.estimators.face_estimators.basic_attributes import (
    BasicAttributesEstimator,
    BasicAttributes,
    Ethnicities,
)
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage, FaceWarper, FaceWarp
from lunavl.sdk.faceengine.setting_provider import DetectorType
from lunavl.sdk.image_utils.image import VLImage
from lunavl.sdk.faceengine.engine import VLFaceEngine
from tests.base import BaseTestClass
from tests.resources import ONE_FACE


def generateWarp(faceEngine: VLFaceEngine, imagePath: Optional[str] = ONE_FACE) -> FaceWarp:
    """
    Generate warps.

    Args:
        faceEngine: Face Engine instance
        imagePath: path to image

    Returns:
        first 1000 warps sorted by coordinates
    """
    detector = faceEngine.createFaceDetector(DetectorType.FACE_DET_V3)
    warper: FaceWarper = faceEngine.createFaceWarper()

    img = VLImage.load(filename=imagePath)
    detection = detector.detect(images=[img], limit=1000, detect68Landmarks=True)[0][0]

    warp = warper.warp(detection)
    return warp


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

    _warp: FaceWarp

    @classmethod
    def setUpClass(cls) -> None:
        """ Load warps. """
        cls._warp = generateWarp(cls.faceEngine)

    def estimate(
        self,
        warp: Union[FaceWarpedImage, FaceWarp],
        estimateAge: bool = False,
        estimateGender: bool = False,
        estimateEthnicity: bool = False,
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
        warps: Union[List[FaceWarpedImage], List[FaceWarp]],
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
        expectedEstimation = Estimation(20, 0, Eth(0, 0, 1, 0))
        for estimationType, delta in (("Age", AGE_DELTA), ("Gender", GENDER_DELTA), ("Ethnicity", ETH_DELTA)):
            estimationFlag = f"estimate{estimationType}"
            basicAttributeGetter: Callable[[BasicAttributes], Union[Ethnicities, float, None]] = attrgetter(
                estimationType.lower()
            )
            expectedAttr = getattr(expectedEstimation, estimationType)
            with self.subTest(estimationType=estimationType):
                singleAttr = basicAttributeGetter(self.estimate(self._warp, **{estimationFlag: True}))
                batchAttr = basicAttributeGetter(self.estimateBatch([self._warp], **{estimationFlag: True})[0][0])
                self.assertNotIn(None, (singleAttr, batchAttr))

                filename = f"./{time()}.jpg"
                msg = f"Batch estimation '{estimationType}' differs from single one. Saved as '{filename}'."
                if isinstance(singleAttr, Ethnicities):
                    self.assertEqual(singleAttr.asian, batchAttr.asian, msg)
                    self.assertEqual(singleAttr.indian, batchAttr.indian, msg)
                    self.assertEqual(singleAttr.caucasian, batchAttr.caucasian, msg)
                    self.assertEqual(singleAttr.africanAmerican, batchAttr.africanAmerican, msg)
                else:
                    self.assertEqual(singleAttr, batchAttr, msg)

                self.assertAlmostEqual(expectedAttr, singleAttr, delta=delta)
