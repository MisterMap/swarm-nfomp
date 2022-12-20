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
