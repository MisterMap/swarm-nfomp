from dataclasses import dataclass

import numpy as np


@dataclass
class Point2D:
    x: float
    y: float

    def as_numpy(self):
        return np.array([self.x, self.y])

    def distance(self, other):
        return np.linalg.norm(self.as_numpy() - other.as_numpy())

    @classmethod
    def from_vec(cls, vec):
        return cls(vec[0], vec[1])

    @classmethod
    def from_dict(cls, d):
        return cls(d["x"], d["y"])
