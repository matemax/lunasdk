"""
Module contains geometric structures (Rect, Point, Size)
"""
from typing import TypeVar, Generic, Union, List, Optional, Dict, Tuple

from FaceEngine import Vector2i, Vector2f  # pylint: disable=E0611,E0401
from FaceEngine import Rect as CoreRectI, RectFloat as CoreRectF  # pylint: disable=E0611,E0401

from FaceEngine import Landmarks5, Landmarks68, IrisLandmarks, EyelidLandmarks  # pylint: disable=E0611,E0401

from lunavl.sdk.estimators.base_estimation import BaseEstimation

COORDINATE_TYPE = TypeVar("COORDINATE_TYPE", float, int)  #: generic type for allowed values type of coordinates
LANDMARKS = TypeVar(
    "LANDMARKS", Landmarks5, Landmarks68, IrisLandmarks, EyelidLandmarks
)  #: generic type for allowed values type of landmarks


class Size(Generic[COORDINATE_TYPE]):
    """
    Rect size.

    Attributes:
        width (COORDINATE_TYPE): width
        height (COORDINATE_TYPE): height
    """

    __slots__ = ("width", "height")

    def __init__(self, width: COORDINATE_TYPE, height: COORDINATE_TYPE):
        """
        Init

        Args:
            width: width
            height: height
        """
        self.width: COORDINATE_TYPE = width
        self.height: COORDINATE_TYPE = height

    def __repr__(self) -> str:
        """
        Size representation.

        Returns:
            "width = {self.width}, height = {self.height}"
        >>> str(Size(1, 2))
        'width = 1, height = 2'
        """
        return "width = {}, height = {}".format(self.width, self.height)

    def asDict(self) -> dict:
        """
        Convert to dict

        Returns:
            {"width": self.width, "height": self.height}
        >>> Size(1, 2).asDict()
        {'width': 1, 'height': 2}
        >>> Size(1.0, 2.0).asDict()
        {'width': 1.0, 'height': 2.0}
        """
        return {"width": self.width, "height": self.height}


class Point(Generic[COORDINATE_TYPE]):
    """
    Point.

    Attributes:
        x (CoordinateType): x-coordinate
        y (CoordinateType): y-coordinate
    """

    __slots__ = ["x", "y"]

    def __init__(self, x: COORDINATE_TYPE, y: COORDINATE_TYPE):  # pylint: disable=C0103
        """
        Init

        Args:
            x: x
            y: y
        """
        self.x: COORDINATE_TYPE = x  # pylint: disable=C0103
        self.y: COORDINATE_TYPE = y  # pylint: disable=C0103

    @classmethod
    def fromVector2(cls, vec2: Union[Vector2f, Vector2i]) -> "Point":
        """
        Create Point from Core Vector2i and Vector2f

        Args:
            vec2: vector2i or vector2f

        Returns:
            point
        """
        point = Point(0, 0)
        point.x = vec2.x
        point.y = vec2.y
        return point

    def toVector2(self) -> Union[Vector2i, Vector2f]:
        """
        Create Vector2i or Vector2f from point

        Returns:
            Vector2i if x and y are integer otherwise Vector2f
        >>> vec2 = Point(1, 2).toVector2()
        >>> isinstance(vec2, Vector2i)
        True
        >>> vec2 = Point(1.0, 2.0).toVector2()
        >>> isinstance(vec2, Vector2f)
        True
        """
        if isinstance(self.x, int) and isinstance(self.y, int):
            return Vector2i(self.x, self.y)
        return Vector2f(self.x, self.y)

    def asDict(self) -> Tuple[COORDINATE_TYPE, COORDINATE_TYPE]:
        """
        Convert point to list

        Returns:
            [self.x, self.y]

        >>> Point(1, 2).asDict()
        (1, 2)
        >>> Point(1.0, 2.0).asDict()
        (1.0, 2.0)
        """
        return self.x, self.y

    def __repr__(self):
        return "x = {}, y = {}".format(self.x, self.y)


class Rect(Generic[COORDINATE_TYPE]):
    """
    Rect

    Attributes:
        coreRect (CoreRect): core rect object

    """

    def __init__(
        self,
        x: COORDINATE_TYPE = 0,
        y: COORDINATE_TYPE = 0,  # pylint: disable=C0103
        width: COORDINATE_TYPE = 0,
        height: COORDINATE_TYPE = 0,
    ):
        """
        Init. If there are argument of type float coreRect will be type CoreRectF otherwise CoreRectI.

        Args:
            x: x
            y: y
            width: width
            height: height
        """
        if any((isinstance(x, float), isinstance(y, float), isinstance(width, float), isinstance(height, float))):
            self.coreRect = CoreRectF(x, y, width, height)
        else:
            self.coreRect = CoreRectI(x, y, width, height)

    @classmethod
    def fromCoreRect(cls, rect: Union[CoreRectF, CoreRectI]) -> "Rect":
        """
        Load rect from core rect

        Args:
            rect: core rect

        Returns:
            new rect
        """
        newRect = cls()
        newRect.coreRect = rect
        return newRect

    @classmethod
    def initByCorners(cls, topLeftCorner: Point[COORDINATE_TYPE], bottomRightBottom: Point[COORDINATE_TYPE]) -> "Rect":
        """
        Init rect by top left corner, bottom right bottom

        Args:
            topLeftCorner: top left corner
            bottomRightBottom: bottom right bottom

        Returns:
            new rect
        """
        newRect = cls()
        newRect.coreRect = newRect.coreRect.set(topLeftCorner.toVector2(), bottomRightBottom.toVector2())
        return newRect

    @property
    def x(self) -> COORDINATE_TYPE:  # pylint: disable=C0103
        """
        Getter of x coordinate

        Returns:
            self._rect.x
        """
        return self.coreRect.x

    @x.setter
    def x(self, value: COORDINATE_TYPE):  # pylint: disable=C0103
        """
        Setter of x

        Args:
            value: new value
        """
        self.coreRect.x = value

    @property
    def y(self) -> COORDINATE_TYPE:  # pylint: disable=C0103
        """
        Getter of y coordinate

        Returns:
            self._rect.y
        """
        return self.coreRect.y

    @y.setter
    def y(self, value: COORDINATE_TYPE):  # pylint: disable=C0103
        """
        Setter of y

        Args:
            value: new value
        """
        self.coreRect.y = value

    @property
    def width(self) -> COORDINATE_TYPE:
        """
        Getter of width

        Returns:
            self._rect.width
        """
        return self.coreRect.width

    @width.setter
    def width(self, value: COORDINATE_TYPE):
        """
        Setter of width

        Args:
            value: new value
        """
        self.coreRect.width = value

    @property
    def height(self) -> COORDINATE_TYPE:
        """
        Getter of height

        Returns:
            self._rect.height
        """
        return self.coreRect.height

    @height.setter
    def height(self, value: COORDINATE_TYPE):
        """
        Setter of height

        Args:
            value: new value
        """
        self.coreRect.height = value

    @property
    def bottom(self) -> COORDINATE_TYPE:  # real signature unknown; restored from __doc__
        """
        Get lower y-coordinate of the rect

        Returns:
            self.y + self.width

        >>> Rect(1, 2, 3, 4).bottom
        6
        """
        vector = self.coreRect.bottom()
        return vector

    @property
    def bottomRight(self) -> Point[COORDINATE_TYPE]:
        """
        Get coordinates of the right bottom angle

        Returns:
            point

        >>> Rect(1, 2, 3, 4).bottomRight
        x = 4, y = 6
        """
        vector = self.coreRect.bottomRight()
        return Point.fromVector2(vector)

    @property
    def top(self) -> COORDINATE_TYPE:
        """
        Get upper y-coordinate of the rect

        Returns:
            self.y

        >>> Rect(1, 2, 3, 4).top
        2

        >>> Rect(1, 2, 3, -4).top
        2
        """
        vector = self.coreRect.top()
        return vector

    @property
    def topLeft(self) -> Point[COORDINATE_TYPE]:  # real signature unknown; restored from __doc__
        """
        Get coordinates of the top left angle

        Returns:
            point

        >>> Rect(1, 2, 3, 4).topLeft
        x = 1, y = 2
        """
        vector = self.coreRect.topLeft()
        return Point.fromVector2(vector)

    @property
    def left(self) -> COORDINATE_TYPE:
        """
        Get lower x-coordinate of the rect

        Returns:
            self.x

        >>> Rect(1, 2, 3, 4).left
        1

        >>> Rect(1, 2, -3, 4).left
        1
        """
        return self.coreRect.left()

    @property
    def right(self) -> COORDINATE_TYPE:
        """
        Get upper x-coordinate of the rect

        Returns:
            self.x

        >>> Rect(1, 2, 3, 4).right
        4

        >>> Rect(1, 2, -3, 4).right
        -2
        """
        return self.coreRect.right()

    @property
    def center(self) -> Point[COORDINATE_TYPE]:
        """
        Get coordinates of the center

        Returns:
            point

        >>> Rect(1, 2, 3, 4).center
        x = 2, y = 4
        >>> Rect(1, 2, 3, 5).center
        x = 2, y = 4
        >>> Rect(1, 2, 4, 4).center
        x = 3, y = 4
        >>> Rect(1.0, 2.0, 4.0, 5.0).center
        x = 3.0, y = 4.5
        """

        vector = self.coreRect.center()
        return Point.fromVector2(vector)

    def getArea(self) -> COORDINATE_TYPE:
        """
        Get rect area

        Returns:
            self.width * self.height

        >>> Rect(1, 2, 3, 4).getArea()
        12
        >>> Rect(1.0, 2.0, 3.5, 4.5).getArea()
        15.75
        """
        return self.coreRect.getArea()

    def isInside(self, other: "Rect") -> bool:
        """
        Check other rect is inside in this or not

        Args:
            other: other rect

        Returns:
            true if this inside of the 'other'

        >>> first = Rect(1, 2, 3, 4)
        >>> second = Rect(1, 2, 3, 3)
        >>> first.isInside(second)
        False
        >>> second.isInside(first)
        True
        """
        return self.coreRect.inside(other.coreRect)

    def isValid(self) -> bool:
        """
        Validate width and height of the rect

        Returns:
            True if  width and height > 0 otherwise False

        >>> Rect(1, 2, 3, 4).isValid()
        True
        >>> Rect(1, 2, -3, 3).isValid()
        False
        """
        return self.coreRect.isValid()

    @property
    def size(self) -> Size[COORDINATE_TYPE]:
        """
        Get rect size

        Returns:
            size

        >>> Rect(1, 2, 3, 4).size
        width = 3, height = 4
        """
        return Size(self.width, self.height)

    def __and__(self, other: "Rect[COORDINATE_TYPE]") -> "Rect[COORDINATE_TYPE]":
        """
        Calculate an intersection of rects.

        Args:
            other: other rect

        Returns:
            intersection of rects


        >>> first = Rect(1, 2, 3, 4)
        >>> second = Rect(2, 1, 3, 3)
        >>> first and second
        x = 2, y = 1, width = 2, height = 1
        """
        return self.coreRect and other.coreRect

    def __eq__(self, other: object) -> bool:
        """
        Compare two rect

        Args:
            other: other rect

        Returns:
            True if x, y, width, height other rect are equal x, y, width, height this rect
        Raises:
            NotImplemented: if other type is not Rect


        >>> first = Rect(1, 2, 3, 4)
        >>> second = Rect(1, 2, 3, 3)
        >>> first == second
        False
        >>> third = Rect(1, 2, 3, 4)
        >>> first == third
        True
        """
        if not isinstance(other, Rect):
            raise NotImplementedError
        return self.coreRect == other.coreRect

    def __ne__(self, other: object) -> bool:
        """
        Compare two rect

        Args:
            other: other rect

        Returns:
            True if any of x, y, width, height other rect are not equal corresponding x, y, width, height of this rect


        >>> first = Rect(1, 2, 3, 4)
        >>> second = Rect(1, 2, 3, 3)
        >>> first != second
        True
        >>> third = Rect(1, 2, 3, 4)
        >>> first != third
        False
        """
        if not isinstance(other, Rect):
            return False
        return self.coreRect != other.coreRect

    def adjust(
        self,
        dx: COORDINATE_TYPE,
        dy: COORDINATE_TYPE,
        dw: COORDINATE_TYPE,  # pylint: disable=C0103
        dh: COORDINATE_TYPE,
    ) -> None:
        """
        Adjusts the rect by given amounts.

        Args:
            dx: adjustment for upper left corner x coordinate
            dy: adjustment for upper left corner y coordinate
            dw: adjustment for width
            dh: adjustment for height
        """
        self.coreRect.adjust(dx, dy, dw, dh)

    # pylint: disable=C0103
    def adjusted(self, dx: COORDINATE_TYPE, dy: COORDINATE_TYPE, dw: COORDINATE_TYPE, dh: COORDINATE_TYPE) -> "Rect":
        """
        Copies and adjusts the rect by given amounts.

        Args:
            dx: adjustment for upper left corner x coordinate
            dy: adjustment for upper left corner y coordinate
            dw: adjustment for width
            dh: adjustment for height

        Returns:
            return copy.
        """
        newRect = Rect()
        newRect.coreRect = self.coreRect.adjusted(dx, dy, dw, dh)
        return newRect

    def asDict(self) -> Dict[str, COORDINATE_TYPE]:
        """
        Convert rect to dict

        Returns:
            {"x": self.x, "y": self.y, "width": self.width, "height": self.height}
        >>> Rect(1, 2, 3, 4).asDict()
        {'x': 1, 'y': 2, 'width': 3, 'height': 4}
        >>> Rect(1.0, 2, 3.0, 4.0).asDict()
        {'x': 1.0, 'y': 2.0, 'width': 3.0, 'height': 4.0}
        """
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def __repr__(self) -> str:
        """
        Dumps rect to string
        Returns:
            "x = {self.x}, y = {self.y}, width = {self.width}, width = {self.height}"

        >>> "{}".format(Rect(1, 2, 3, 4))
        'x = 1, y = 2, width = 3, height = 4'
        >>> "{}".format(Rect(1.0, 2.0, 3.0, 4.0))
        'x = 1.000000, y = 2.000000, width = 3.000000, height = 4.000000'

        """
        return self.coreRect.__repr__()


class Landmarks(BaseEstimation):
    """
    Base class for landmarks

    Attributes:
        _points (Optional[List[Point[float]]]): lazy load attributes, converted to point list core landmarks
    """

    __slots__ = ["_points", "_coreLandmarks"]

    def __init__(self, coreLandmarks: LANDMARKS):
        """
        Init

        Args:
            coreLandmarks (LANDMARKS): core landmarks
        """
        super().__init__(coreLandmarks)
        self._points: Optional[List[Point[float]]] = None

    @property
    def points(self) -> Tuple[Point[float], ...]:
        """
        Lazy load of points.

        Returns:
            list of points
        """
        if self._points is None:
            self._points = tuple((Point.fromVector2(self._coreEstimation[index]) for index in
                                  range(len(self._coreEstimation))))
        return self._points

    def asDict(self) -> Tuple[Tuple[float, float], ...]:
        """
        Convert to dict

        Returns:
            list to list points
        """
        pointCount = len(self._coreEstimation)
        points = self.coreEstimation
        return tuple(((points[index].x, points[index].y) for index in range(pointCount)))
