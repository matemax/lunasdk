from collections import namedtuple
from dataclasses import dataclass, asdict
from operator import attrgetter, itemgetter
from statistics import mean

import pytest
from time import time
from typing import List, Union, Callable, Tuple

import jsonschema

from lunavl.sdk.errors.errors import LunaVLError
from lunavl.sdk.errors.exceptions import LunaSDKException
from lunavl.sdk.estimators.face_estimators.basic_attributes import (
    BasicAttributesEstimator,
    BasicAttributes,
    Ethnicities,
    Ethnicity,
)
from lunavl.sdk.estimators.face_estimators.facewarper import FaceWarpedImage, FaceWarp
from lunavl.sdk.image_utils.image import VLImage
from tests.base import BaseTestClass
from tests.detect_test_class import VLIMAGE_SMALL
from tests.resources import WARP_ONE_FACE, WARP_CLEAN_FACE


@dataclass
class Eth:
    """ Class for Ethnicities estimation. """

    class Predominant(str):
        def __eq__(self, other):
            if isinstance(other, Ethnicity):
                return str(other) == self
            return super().__eq__(other)

    asian: float
    indian: float
    caucasian: float
    africanAmerican: float

    @property
    def predominantEthnicity(self) -> Predominant:
        ethnicities = list(asdict(self).items())
        ethnicities.sort(key=itemgetter(1), reverse=True)
        return self.Predominant(ethnicities[0][0])

    @classmethod
    def fromSeveral(cls, eths: List[Ethnicities]):
        args = ("asian", "indian", "caucasian", "africanAmerican")
        kwargs = {arg: mean(map(attrgetter(arg), eths)) for arg in args}
        return cls(**kwargs)


Estimation = namedtuple("Estimation", ("Age", "Gender", "Ethnicity"))


class TestBasicAttributes(BaseTestClass):
    """ Test basic attributes. """

    # estimator to call
    estimator: BasicAttributesEstimator = BaseTestClass.faceEngine.createBasicAttributesEstimator()

    # warped image
    _warp: FaceWarpedImage
    # second warped image
    _warp2: FaceWarpedImage

    @classmethod
    def setUpClass(cls) -> None:
        """ Load warps. """
        cls._warp = FaceWarpedImage.load(filename=WARP_ONE_FACE)
        cls._warp2 = FaceWarpedImage.load(filename=WARP_CLEAN_FACE)

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
        warps: List[Union[FaceWarp, FaceWarpedImage]],
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

    @staticmethod
    def assertEth(eth1: Union[Ethnicities, Eth], eth2: Union[Ethnicities, Eth], delta: float = 0.003) -> None:
        """
        Assert ethnicities.

        Args:
            eth1: first ethnicities
            eth2: second ethnicities
            delta: allowed delta
        """
        for attrName in ("asian", "indian", "caucasian", "africanAmerican"):
            fstAttr, sndAttr = map(attrgetter(attrName), (eth1, eth2))
            assert abs(fstAttr - sndAttr) <= delta, f"Attribute '{attrName}' differ: {fstAttr} {sndAttr}"
        assert eth1.predominantEthnicity == eth2.predominantEthnicity

    def test_correctness(self):
        """
        Test estimation correctness.
        """
        expectedEstimation = Estimation(20, 0, Eth(0, 0, 1, 0))
        for estimationType in ("Age", "Gender", "Ethnicity"):
            estimationFlag = f"estimate{estimationType}"
            basicAttributeGetter: Callable[[BasicAttributes], Union[Ethnicities, float, None]] = attrgetter(
                estimationType.lower()
            )
            expectedValue = getattr(expectedEstimation, estimationType)
            with self.subTest(estimationType=estimationType):
                singleValue = basicAttributeGetter(self.estimate(self._warp, **{estimationFlag: True}))
                batchValue = basicAttributeGetter(self.estimateBatch([self._warp] * 2, **{estimationFlag: True})[0][0])
                assert type(singleValue) == type(batchValue)
                assert isinstance(singleValue, (float, Ethnicities))

                filename = f"./{time()}.jpg"
                msg = f"Batch estimation '{estimationType}' differs from single one. Saved as '{filename}'."
                if isinstance(singleValue, Ethnicities):
                    self.assertEth(expectedValue, singleValue)
                    self.assertEth(expectedValue, batchValue)
                else:  # age or gender
                    assert expectedValue == int(singleValue) == int(batchValue), msg

    def test_aggregation(self):
        """
        Test aggregation correctness.
        """
        estimations, aggregatedEstimation = self.estimateBatch(
            [self._warp, self._warp2], estimateAge=True, estimateGender=True, estimateEthnicity=True, aggregate=True
        )
        for estimationType in ("Age", "Gender", "Ethnicity"):
            estimationGetter: Callable[[Union[BasicAttributes, Eth]], Union[Ethnicities, float]] = attrgetter(
                estimationType.lower()
            )
            with self.subTest(estimationType):
                raw = list(map(estimationGetter, estimations))
                aggregated = estimationGetter(aggregatedEstimation)
                if isinstance(aggregated, Ethnicities):
                    self.assertEth(Eth.fromSeveral(raw), aggregated)
                else:  # age or gender
                    assert int(mean(raw)) == int(aggregated)

    def test_as_dict(self):
        """
        Test asDict method.
        """
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "maximum": 100, "minimum": 0},
                "gender": {"type": "integer", "maximum": 1, "minimum": 0},
                "ethnicities": {
                    "type": "object",
                    "properties": {
                        "predominant_ethnicity": {"type": "string"},
                        "estimations": {
                            "properties": {
                                "asian": {"type": "number", "maximum": 1, "minimum": 0},
                                "indian": {"type": "number", "maximum": 1, "minimum": 0},
                                "caucasian": {"type": "number", "maximum": 1, "minimum": 0},
                                "african_american": {"type": "number", "maximum": 1, "minimum": 0},
                            },
                            "required": ["asian", "indian", "caucasian", "african_american"],
                        },
                    },
                    "required": ["predominant_ethnicity", "estimations"],
                },
            },
            "required": ["ethnicities", "age", "gender"],
        }
        raw, aggregated = self.estimateBatch(
            [self._warp, self._warp2], estimateAge=True, estimateGender=True, estimateEthnicity=True, aggregate=True
        )
        for estimation in [*raw, aggregated]:
            jsonschema.validate(estimation.asDict(), schema)
            assert isinstance(estimation.asDict(), dict)

    def test_batch_estimate_with_success_and_error(self):
        """
        Test batch estimate with good and bad warp.
        """
        badWarp = FaceWarpedImage(VLImage.load(filename=WARP_CLEAN_FACE))
        badWarp.coreImage = VLIMAGE_SMALL.coreImage
        with pytest.raises(LunaSDKException) as exceptionInfo:
            self.estimator.estimateBasicAttributesBatch(
                warps=[self._warp, badWarp],
                estimateAge=True,
                estimateGender=True,
                estimateEthnicity=True,
                aggregate=False,
            )
        self.assertLunaVlError(exceptionInfo, LunaVLError.BatchedInternalError.format("Failed validation."))
        assert len(exceptionInfo.value.context) == 2, "Expect two errors in exception context"
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[0], LunaVLError.Ok)
        self.assertReceivedAndRawExpectedErrors(exceptionInfo.value.context[1], LunaVLError.InvalidImageSize)

    def test_async_estimate_basic_attributes(self):
        """
        Test async estimate basic attributes
        """
        task = self.estimator.estimate(
            warp=self._warp, estimateAge=True, estimateGender=True, estimateEthnicity=True, asyncEstimate=True
        )
        self.assertAsyncEstimation(task, BasicAttributes)
        task = self.estimator.estimateBasicAttributesBatch(
            warps=[self._warp] * 2,
            estimateAge=True,
            estimateGender=True,
            estimateEthnicity=True,
            asyncEstimate=True,
            aggregate=False,
        )
        self.assertAsyncBatchEstimation(task, BasicAttributes)

        task = self.estimator.estimateBasicAttributesBatch(
            warps=[self._warp] * 2,
            estimateAge=True,
            estimateGender=True,
            estimateEthnicity=True,
            asyncEstimate=True,
            aggregate=True,
        )
        self.assertAsyncBatchEstimationWithAggregation(task, BasicAttributes)
