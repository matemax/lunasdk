"""
Module contains geometric structures (Rect, Point, Size)
"""
from typing import TypeVar, Generic, Union, List

from FaceEngine import Vector2i, Vector2f  # pylint: disable=E0611,E0401
from FaceEngine import Rect as CoreRectI, RectFloat as CoreRectF  # pylint: disable=E0611,E0401

CoordinateType = TypeVar('CoordinateType', float, int)  #: generic type for allowed values type of coordinates


class Size(Generic[CoordinateType]):
    """
    Rect size.

    Attributes:
        width (CoordinateType): width
        height (CoordinateType): height
    """
    __slots__ = ["width", "height"]

    def __init__(self, width: CoordinateType, height: CoordinateType):
        """
        Init

        Args:
            width: width
            height: height
        """
        self.width = width
        self.height = height

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


class Point(Generic[CoordinateType]):
    """
    Point.

    Attributes:
        x (CoordinateType): x-coordinate
        y (CoordinateType): y-coordinate
    """
    __slots__ = ["x", "y"]

    def __init__(self, x: CoordinateType, y: CoordinateType):
        """
        Init

        Args:
            x: x
            y: y
        """
        self.x = x  # pylint: disable=C0103
        self.y = y  # pylint: disable=C0103

    @classmethod
    def fromVector2(cls, vec2: Union[Vector2f, Vector2i]) -> 'Point':
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

    def asDict(self) -> List[CoordinateType]:
        """
        Convert point to list

        Returns:
            [self.x, self.y]

        >>> Point(1, 2).asDict()
        [1, 2]
        >>> Point(1.0, 2.0).asDict()
        [1.0, 2.0]
        """
        return [self.x, self.y]

    def __repr__(self):
        return "x = {}, y = {}".format(self.x, self.y)


class Rect(Generic[CoordinateType]):
    """
    Rect

    Attributes:
        coreRect (CoreRect): core rect object

    """

    def __init__(self, x: CoordinateType = 0, y: CoordinateType = 0,
                 width: CoordinateType = 0, height: CoordinateType = 0):
        """
        Init. If there are argument of type float coreRect will be type CoreRectF otherwise CoreRectI.

        Args:
            x: x
            y: y
            width: width
            height: height
        """
        if any((isinstance(x, float), isinstance(y, float),
                isinstance(width, float), isinstance(height, float))):
            self.coreRect = CoreRectF(x, y, width, height)
        else:
            self.coreRect = CoreRectI(x, y, width, height)

    @classmethod
    def fromCoreRect(cls, rect: Union[CoreRectF, CoreRectI]):
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

    @property
    def x(self) -> CoordinateType:  # pylint: disable=C0103
        """
        Getter of x coordinate

        Returns:
            self._rect.x
        """
        return self.coreRect.x

    @x.setter
    def x(self, value: CoordinateType):  # pylint: disable=C0103
        """
        Setter of x

        Args:
            value: new value
        """
        self.coreRect.x = value

    @property
    def y(self) -> CoordinateType:  # pylint: disable=C0103
        """
        Getter of y coordinate

        Returns:
            self._rect.y
        """
        return self.coreRect.y

    @y.setter
    def y(self, value: CoordinateType):  # pylint: disable=C0103
        """
        Setter of y

        Args:
            value: new value
        """
        self.coreRect.y = value

    @property
    def width(self) -> CoordinateType:
        """
        Getter of width

        Returns:
            self._rect.width
        """
        return self.coreRect.width

    @width.setter
    def width(self, value: CoordinateType):
        """
        Setter of width

        Args:
            value: new value
        """
        self.coreRect.width = value

    @property
    def height(self) -> CoordinateType:
        """
        Getter of height

        Returns:
            self._rect.height
        """
        return self.coreRect.height

    @height.setter
    def height(self, value: CoordinateType):
        """
        Setter of height

        Args:
            value: new value
        """
        self.coreRect.height = value

    @property
    def bottom(self) -> CoordinateType:  # real signature unknown; restored from __doc__
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
    def bottomRight(self) -> Point[CoordinateType]:
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
    def top(self) -> CoordinateType:
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
    def topLeft(self) -> Point[CoordinateType]:  # real signature unknown; restored from __doc__
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
    def left(self) -> CoordinateType:
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
    def right(self) -> CoordinateType:
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
    def center(self) -> Point[CoordinateType]:
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

    def getArea(self) -> CoordinateType:
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

    def isInside(self, other: 'Rect') -> bool:
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
    def size(self) -> Size[CoordinateType]:
        """
        Get rect size

        Returns:
            size

        >>> Rect(1, 2, 3, 4).size
        width = 3, height = 4
        """
        return Size(self.width, self.height)

    def __and__(self, other: 'Rect[CoordinateType]') -> 'Rect[CoordinateType]':
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

    def __eq__(self, other: 'Rect[CoordinateType]') -> bool:
        """
        Compare two rect

        Args:
            other: other rect

        Returns:
            True if x, y, width, height other rect are equal x, y, width, height this rect


        >>> first = Rect(1, 2, 3, 4)
        >>> second = Rect(1, 2, 3, 3)
        >>> first == second
        False
        >>> third = Rect(1, 2, 3, 4)
        >>> first == third
        True
        """
        return self.coreRect == other.coreRect

    def __ne__(self, other: 'Rect[CoordinateType]') -> bool:
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
        return self.coreRect != other.coreRect

    def adjust(self, arg0, arg1, arg2, arg3):  # real signature unknown; restored from __doc__
        """ adjust(self: FaceEngine.Rect, arg0: int, arg1: int, arg2: int, arg3: int) -> None """
        pass

    def adjusted(self, arg0, arg1, arg2, arg3):  # real signature unknown; restored from __doc__
        """ adjusted(self: FaceEngine.Rect, arg0: int, arg1: int, arg2: int, arg3: int) -> FaceEngine.Rect """
        pass

    def set(self, arg0, *args, **kwargs):  # real signature unknown; NOTE: unreliably restored from __doc__
        """ set(self: FaceEngine.Rect, arg0: fsdk::Vector2<int>, arg1: fsdk::Vector2<int>) -> None """
        pass

    def coords(self, arg0, arg1, arg2, arg3):  # real signature unknown; restored from __doc__
        """ coords(arg0: int, arg1: int, arg2: int, arg3: int) -> FaceEngine.Rect """
        pass

    def asDict(self) -> dict:
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
