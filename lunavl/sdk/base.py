from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Dict, Any

from FaceEngine import DetectionFloat, HumanLandmark, HumanLandmarks17  # pylint: disable=E0611,E0401

from .image_utils.geometry import LANDMARKS, Point, Rect


class BaseEstimation(ABC):
    """
    Base class for estimation structures.

    Attributes:
        _coreEstimation: core estimation
    """

    __slots__ = ("_coreEstimation",)

    def __init__(self, coreEstimation: Any):
        self._coreEstimation = coreEstimation

    @property
    def coreEstimation(self):
        """
        Get core estimation from init
        Returns:
            _coreEstimation
        """
        return self._coreEstimation

    @abstractmethod
    def asDict(self) -> Union[dict, list]:
        """
        Convert to a dict.

        Returns:
            dict from luna api
        """
        pass

    def __repr__(self) -> str:
        """
        Representation.

        Returns:
            str(self.asDict())
        """
        return str(self.asDict())


class Landmarks(BaseEstimation):
    """
    Base class for landmarks

    Attributes:
        _points (Optional[Tuple[Point[float]]]): lazy load attributes, converted to point list core landmarks
    """

    __slots__ = ["_points", "_coreEstimation"]

    def __init__(self, coreLandmarks: LANDMARKS):
        """
        Init

        Args:
            coreLandmarks (LANDMARKS): core landmarks
        """
        super().__init__(coreLandmarks)
        self._points: Optional[Tuple[Point[float], ...]] = None

    @property
    def points(self) -> Tuple[Point[float], ...]:
        """
        Lazy load of points.

        Returns:
            list of points
        """
        if self._points is None:
            self._points = tuple(
                (Point.fromVector2(self._coreEstimation[index]) for index in range(len(self._coreEstimation)))
            )
        return self._points

    def asDict(self) -> Tuple[Tuple[int, int], ...]:  # type: ignore
        """
        Convert to dict

        Returns:
            list to list points
        """
        pointCount = len(self._coreEstimation)
        points = self._coreEstimation
        return tuple(((int(points[index].x), int(points[index].x)) for index in range(pointCount)))


class LandmarkWithScore(BaseEstimation):
    """
    Point with score.
    """

    def __init__(self, landmark: HumanLandmark):  # pylint: disable=C0103
        """
        Init

        Args:
            landmark: core landmark
        """
        super().__init__(landmark)

    @property
    def point(self) -> Point[float]:
        """
        Coordinate of landmark
        Returns:
            point
        """
        return Point.fromVector2(self._coreEstimation.point)

    @property
    def score(self) -> float:
        """
        Landmark score
        Returns:
            float[0,1]
        """
        return self._coreEstimation.score

    def asDict(self) -> dict:
        """
        Convert point to list (json),  coordinates will be cast from float to int

        Returns:
            dict with keys: score and point
        """
        return {"score": self._coreEstimation.score, "point": (int(self.point.x), int(self.point.y))}

    def __repr__(self) -> str:
        """
        Representation.

        Returns:
            "x = {self.point.x}, y = {self.point.y}, score = {self.score}"
        """
        return "x = {}, y = {}, score = {}".format(self.point.x, self.point.y, self.score)


class LandmarksWithScore(BaseEstimation):
    """
    Base class for landmarks with score

    Attributes:
        _points (Optional[Tuple[Point[float]]]): lazy load attributes, converted to point list core landmarks
    """

    __slots__ = ["_points", "_coreEstimation"]

    def __init__(self, coreLandmarks: HumanLandmarks17):
        """
        Init

        Args:
            coreLandmarks (LANDMARKS): core landmarks
        """
        super().__init__(coreLandmarks)
        self._points: Optional[Tuple[LandmarkWithScore, ...]] = None

    @property
    def points(self) -> Tuple[LandmarkWithScore, ...]:
        """
        Lazy load of points.

        Returns:
            list of points
        """
        if self._points is None:
            self._points = tuple(
                (LandmarkWithScore(self._coreEstimation[index]) for index in range(len(self._coreEstimation)))
            )
        return self._points

    def asDict(self) -> Tuple[dict, ...]:  # type: ignore
        """
        Convert to dict

        Returns:
            list to list points
        """
        return tuple(point.asDict() for point in self.points)


class BoundingBox(BaseEstimation):
    """
    Detection bounding box, it is characterized of rect and score:

        - rect (Rect[float]): face bounding box
        - score (float): face score (0,1), detection score is the measure of classification confidence
                         and not the source image quality. It may be used topick the most "*confident*" face of many.
    """

    #  pylint: disable=W0235
    def __init__(self, boundingBox: DetectionFloat):
        """
        Init.

        Args:
            boundingBox: core bounding box
        """
        super().__init__(boundingBox)

    @property
    def score(self) -> float:
        """
        Get score

        Returns:
            number in range [0,1]
        """
        return self._coreEstimation.score

    @property
    def rect(self) -> Rect[float]:
        """
        Get rect.

        Returns:
            float rect
        """
        return Rect.fromCoreRect(self._coreEstimation.rect)

    def asDict(self) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Convert to  dict.

        Returns:
            {"rect": self.rect, "score": self.score}
        """
        return {"rect": self.rect.asDict(), "score": self.score}
