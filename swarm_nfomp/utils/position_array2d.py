from typing import List

import numpy as np


from swarm_nfomp.utils.position2d import Position2D


class PositionArray2D(Position2D):
    @classmethod
    def from_vec(cls, vec: np.ndarray):
        assert isinstance(vec, np.ndarray)
        assert vec.shape[-1] == 3
        assert len(vec.shape) == 2
        return cls(vec[:, 0], vec[:, 1], vec[:, 2])

    @classmethod
    def from_list(cls, positions: List[Position2D]):
        x = np.array([x_.x for x_ in positions])
        y = np.array([x_.y for x_ in positions])
        angle = np.array([x_.rotation for x_ in positions])
        return cls(x, y, angle)

    def as_list(self) -> List[Position2D]:
        return [Position2D(x, y, theta) for x, y, theta in zip(self._x, self._y, self._angle)]

    def __getitem__(self, item):
        if isinstance(item, slice):
            return PositionArray2D(self._x[item], self._y[item], self._angle[item])
        return Position2D(self._x[item], self._y[item], self._angle[item])

    def __len__(self):
        return self._x.shape[0]

    def __mul__(self, other):
        x1, y1, a1 = self._mul_impl(other)
        return self.__class__(x1, y1, a1)

    def __repr__(self):
        return f"PositionArray2D(x={self.x}, y={self.y}, angle={self.rotation})"
