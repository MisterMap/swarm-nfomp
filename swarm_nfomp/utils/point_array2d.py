class PointArray2D:
    def __init__(self, x, y):
        assert x.shape == y.shape
        self._x = x
        self._y = y

    def as_numpy(self):
        return np.stack([self._x, self._y], axis=-1)

    @classmethod
    def from_vec(cls, vec):
        return cls(vec[..., 0], vec[..., 1])

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __repr__(self):
        return f"PointArray2D(x={self.x}, y={self.y})"
