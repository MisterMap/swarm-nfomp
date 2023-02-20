import numpy as np


class Position2D:
    def __init__(self, x, y, angle):
        self._x: np.ndarray = x
        self._y: np.ndarray = y
        self._angle: np.ndarray = angle

    @property
    def rotation(self):
        return self._angle

    @property
    def translation(self):
        return np.array([self._x, self._y]).T

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def angle(self):
        return self._angle

    @classmethod
    def from_vec(cls, vec):
        assert isinstance(vec, np.ndarray)
        assert vec.shape[-1] == 3
        assert len(vec.shape) == 1
        return cls(vec[0], vec[1], vec[2])

    def as_vec(self):
        return np.array([self._x, self._y, self._angle]).T

    def _mul_impl(self, other):
        x1 = other.x * np.cos(self._angle) - other.y * np.sin(self._angle) + self._x
        y1 = other.x * np.sin(self._angle) + other.y * np.cos(self._angle) + self._y
        a1 = (other.rotation + self._angle + np.pi) % (2 * np.pi) - np.pi
        return x1, y1, a1

    def __mul__(self, other):
        x1, y1, a1 = self._mul_impl(other)
        return other.__class__(x1, y1, a1)

    def inv(self):
        x = -self.x * np.cos(self.rotation) - self.y * np.sin(self.rotation)
        y = self.x * np.sin(self.rotation) - self.y * np.cos(self.rotation)
        return self.__class__(x, y, -self.rotation)

    def apply(self, points: np.ndarray):
        assert points.shape[-1] == 2
        x = points[..., 0]
        y = points[..., 1]
        x1 = x * np.cos(self._angle) - y * np.sin(self._angle) + self._x
        y1 = x * np.sin(self._angle) + y * np.cos(self._angle) + self._y
        return np.stack([x1, y1], axis=-1)

    def __repr__(self):
        return f"Position2D(x={self._x}, y={self._y}, angle={self._angle})"

    def distance(self, position):
        return np.sqrt((position.x - self.x) ** 2 + (position.y - self.y) ** 2)

    def __eq__(self, other):
        return (np.all(self.x == other.x)) and (np.all(self.y == other.y)) and (np.all(self.rotation == other.rotation))

    def as_matrix(self):
        return np.array([[np.cos(self.rotation), -np.sin(self.rotation), self.x],
                         [np.sin(self.rotation), np.cos(self.rotation), self.y],
                         [0, 0, 1]])
