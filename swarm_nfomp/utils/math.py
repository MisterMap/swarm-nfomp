import numpy as np
import torch


def wrap_angles(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def interpolate_1d_pytorch(x: torch.Tensor, old_times: torch.Tensor, new_times: torch.Tensor) -> torch.Tensor:
    upper_indices: torch.Tensor = torch.searchsorted(old_times, new_times)
    lower_indices: torch.Tensor = upper_indices - 1
    lower_indices = torch.where(lower_indices > 0, lower_indices, 0)
    upper_indices = torch.where(upper_indices < x.shape[0], upper_indices, x.shape[0] - 1)
    lower_old_times = torch.gather(old_times, 0, lower_indices)
    upper_old_times = torch.gather(old_times, 0, upper_indices)
    lower_x: torch.Tensor = x[lower_indices]
    upper_x: torch.Tensor = x[upper_indices]
    denominator = upper_old_times - lower_old_times
    denominator = torch.where(torch.abs(denominator) > 1e-6, denominator, torch.full_like(denominator, 1e-6))
    weights = (new_times - lower_old_times) / denominator
    weights = weights[:, None]
    interpolated_values = (1 - weights) * lower_x + weights * upper_x
    return interpolated_values


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


class RectangleRegionArray:
    def __init__(self, min_x: np.ndarray, max_x: np.ndarray, min_y: np.ndarray, max_y: np.ndarray):
        assert min_x.shape == max_x.shape
        assert min_x.shape == min_y.shape
        assert min_x.shape == max_y.shape
        assert np.all(min_x < max_x)
        assert np.all(min_y < max_y)
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def inside(self, positions: PointArray2D) -> np.ndarray:
        result = self.min_x <= positions.x[:, None]
        result &= positions.x[:, None] <= self.max_x
        result &= self.min_y <= positions.y[:, None]
        result &= positions.y[:, None] <= self.max_y
        return np.any(result, axis=1)

    @classmethod
    def from_dict(cls, data):
        data = np.array(data)
        assert len(data.shape) == 2
        assert data.shape[1] == 4
        return cls(data[:, 0], data[:, 1], data[:, 2], data[:, 3])

    def __len__(self):
        return len(self.min_x)
