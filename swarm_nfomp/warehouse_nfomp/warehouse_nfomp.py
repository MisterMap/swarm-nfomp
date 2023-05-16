import dataclasses
from typing import Optional, List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from swarm_nfomp.collision_detector.multi_robot_collision_detector import MultiRobotCollisionDetector
from swarm_nfomp.utils.math import interpolate_1d_pytorch, wrap_angles
from swarm_nfomp.utils.position_array2d import PositionArray2D
from swarm_nfomp.utils.rectangle_bounds import RectangleBounds2D
from swarm_nfomp.utils.timer import Timer


@dataclasses.dataclass
class MultiRobotPathPlannerTask:
    start: PositionArray2D
    goal: PositionArray2D
    collision_detector: MultiRobotCollisionDetector
    bounds: RectangleBounds2D


@dataclasses.dataclass
class MultiRobotResultPath:
    positions_: np.array  # [ n_optimized_states, n_robots, 3]

    @property
    def robot_paths(self) -> List[PositionArray2D]:
        return [PositionArray2D.from_vec(self.positions_[:, i]) for i in range(self.positions_.shape[1])]

    @property
    def numpy_positions(self):
        return self.positions_


@dataclasses.dataclass
class MultiRobotPathOptimizedState:
    positions_: torch.Tensor  # [ n_optimized_states, n_robots, 3]
    direction_constraint_multipliers: torch.Tensor
    start_position: torch.Tensor  # [n_robots, 3]
    goal_position: torch.Tensor  # [n_robots, 3]

    @property
    def result_path(self) -> MultiRobotResultPath:
        return MultiRobotResultPath(
            self.positions.cpu().detach().numpy()
        )

    @property
    def positions(self) -> torch.Tensor:
        return torch.cat(
            [self.start_position[None], self.positions_, self.goal_position[None]], dim=0)

    def reparametrize(self):
        distances = self.calculate_distances()
        old_times = torch.cumsum(distances, dim=0)
        old_times = torch.cat([torch.zeros(1), old_times], dim=0)
        new_times = torch.linspace(0, old_times[-1], old_times.shape[0])
        positions: torch.Tensor = self.positions
        reshaped_positions = positions.reshape(self.positions.shape[0], -1)
        interpolated_positions = interpolate_1d_pytorch(reshaped_positions, old_times, new_times)[1:-1]
        self.positions_.data = interpolated_positions.reshape(*self.positions_.shape)
        multipliers_old_times = (old_times[:-1] + old_times[1:]) / 2
        multipliers_new_times = (new_times[:-1] + new_times[1:]) / 2
        self.direction_constraint_multipliers.data = interpolate_1d_pytorch(
            self.direction_constraint_multipliers, multipliers_old_times, multipliers_new_times)

    def calculate_distances(self) -> torch.Tensor:
        points = self.positions[:, :, :2]
        drone_distances = torch.linalg.norm(points[1:] - points[:-1], dim=2)
        return torch.max(drone_distances, dim=1).values

    @property
    def optimized_parameters(self):
        return [self.positions_]


@dataclasses.dataclass
class OptimizerImplConfig:
    lr: float
    beta1: float
    beta2: float


class OptimizerImpl:
    def __init__(self, parameters: OptimizerImplConfig):
        self._optimizer = None
        self._parameters = parameters

    def setup(self, model_parameters):
        self._optimizer = torch.optim.Adam(model_parameters, lr=self._parameters.lr,
                                           betas=(self._parameters.beta1, self._parameters.beta2))

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self._optimizer.step()


@dataclasses.dataclass
class OptimizerWithLagrangeMultipliersConfig(OptimizerImplConfig):
    lagrange_multiplier_lr: float
    base_lr: float
    max_lr: float
    step_size_up: int
    step_size_down: int


class OptimizerWithLagrangeMultipliers(OptimizerImpl):
    def __init__(self, parameters: OptimizerWithLagrangeMultipliersConfig):
        super().__init__(parameters)
        self._parameters = parameters
        self._lagrange_multiplier_parameters = None
        self._scheduler = None

    # noinspection PyMethodOverriding
    def setup(self, model_parameters, lagrange_multiplier_parameters):
        self._optimizer = torch.optim.RMSprop(model_parameters, lr=self._parameters.lr)
        self._scheduler = torch.optim.lr_scheduler.CyclicLR(self._optimizer, base_lr=self._parameters.base_lr,
                                                            max_lr=self._parameters.max_lr,
                                                            step_size_up=self._parameters.step_size_up,
                                                            step_size_down=self._parameters.step_size_down,
                                                            cycle_momentum=False)
        self._lagrange_multiplier_parameters = lagrange_multiplier_parameters

    def zero_grad(self):
        self._optimizer.zero_grad()
        for p in self._lagrange_multiplier_parameters:
            p.grad = None

    def step(self):
        self._scheduler.step()
        self._optimizer.step()
        with torch.no_grad():
            for p in self._lagrange_multiplier_parameters:
                p.data += self._parameters.lagrange_multiplier_lr * p.grad


class PathOptimizedStateInitializer:
    def __init__(self, planner_task: MultiRobotPathPlannerTask, path_state_count: int, device: str):
        self._planner_task = planner_task
        self._path_state_count = path_state_count
        self._device = device

    def init(self) -> MultiRobotPathOptimizedState:
        with torch.no_grad():
            positions = self._initialize_positions()
            start_position = torch.tensor(self._planner_task.start.as_vec(), requires_grad=False, device=self._device,
                                          dtype=torch.float32)
            goal_position = torch.tensor(self._planner_task.goal.as_vec(), requires_grad=False, device=self._device,
                                         dtype=torch.float32)
            goal_position[:, 2] = start_position[:, 2] + wrap_angles(goal_position[:, 2] - start_position[:, 2])
            return MultiRobotPathOptimizedState(
                positions_=positions[1:-1].clone().detach().requires_grad_(True),
                start_position=start_position,
                goal_position=goal_position,
                direction_constraint_multipliers=torch.zeros(self._path_state_count + 1, len(self._planner_task.start),
                                                             requires_grad=True,
                                                             device=self._device, dtype=torch.float32)
            )

    def _initialize_positions(self) -> torch.Tensor:
        start_point: np.ndarray = self._planner_task.start.as_vec()
        goal_point: np.ndarray = self._planner_task.goal.as_vec()
        trajectory_length = self._path_state_count + 2
        trajectory = torch.zeros(self._path_state_count + 2, len(start_point), 3, requires_grad=True,
                                 device=self._device, dtype=torch.float32)
        for i in range(len(start_point)):
            trajectory[:, i, 0] = torch.linspace(start_point[i, 0], goal_point[i, 0], trajectory_length)
            trajectory[:, i, 1] = torch.linspace(start_point[i, 1], goal_point[i, 1], trajectory_length)
            trajectory[:, i, 2] = start_point[i, 2] + torch.linspace(0,
                                                                     wrap_angles(goal_point[i, 2] - start_point[i, 2]),
                                                                     trajectory_length)
        return trajectory


@dataclasses.dataclass
class PathLossBuilderConfig:
    regularization_weight: float
    collision_weight: float
    direction_constraint_weight: float
    second_differences_weight: float


class MultiRobotPathLossBuilder:
    def __init__(self, planner_task: MultiRobotPathPlannerTask, parameters: PathLossBuilderConfig):
        self._parameters = parameters
        self._planner_task = planner_task

    def get_loss(self, collision_model: nn.Module, optimized_state: MultiRobotPathOptimizedState):
        loss = self._parameters.regularization_weight * self._distance_loss(optimized_state)
        loss = loss + self._parameters.collision_weight * self._collision_loss(collision_model, optimized_state)
        loss = loss + self._direction_constraint_loss(optimized_state)
        loss = loss + self._parameters.second_differences_weight * self._second_differences_loss(optimized_state)
        return loss

    @staticmethod
    def _distance_loss(optimized_state: MultiRobotPathOptimizedState):
        points = optimized_state.positions
        delta = points[1:] - points[:-1]
        delta[:, :, 2] = wrap_angles(delta[:, :, 2])
        return torch.mean(torch.abs(delta) ** 1.5)

    def _collision_loss(self, collision_model: nn.Module, optimized_state: MultiRobotPathOptimizedState):
        positions: torch.Tensor = optimized_state.positions
        positions = self.get_intermediate_points(positions)
        positions = positions.reshape(positions.shape[0], -1)
        return torch.mean(torch.nn.functional.softplus(collision_model(positions)))

    @staticmethod
    def get_intermediate_points(positions):
        t = torch.rand(positions.shape[0] - 1)
        delta = positions[1:] - positions[:-1]
        delta[:, :, 2] = wrap_angles(delta[:, :, 2])
        return positions[1:] + t[:, None, None] * delta

    def _direction_constraint_loss(self, optimized_state: MultiRobotPathOptimizedState):
        deltas = self.non_holonomic_constraint_deltas(optimized_state.positions)
        return self._parameters.direction_constraint_weight * torch.mean(deltas ** 2) + torch.mean(
            optimized_state.direction_constraint_multipliers * deltas)

    @staticmethod
    def non_holonomic_constraint_deltas(positions):
        dx = positions[1:, :, 0] - positions[:-1, :, 0]
        dy = positions[1:, :, 1] - positions[:-1, :, 1]
        angles = positions[:, :, 2]
        delta_angles = wrap_angles(angles[1:] - angles[:-1])
        mean_angles = angles[:-1] + delta_angles / 2
        return dx * torch.sin(mean_angles) - dy * torch.cos(mean_angles)

    @staticmethod
    def _second_differences_loss(optimized_state: MultiRobotPathOptimizedState):
        return torch.mean(
            (optimized_state.positions[2:] - 2 * optimized_state.positions[1:-1] + optimized_state.positions[:-2]) ** 2)


class GradPreconditioner:
    def __init__(self, device, velocity_hessian_weight: float):
        self._device = device
        self._velocity_hessian_weight = velocity_hessian_weight
        self._inverse_hessian = None

    def precondition(self, optimized_state: MultiRobotPathOptimizedState):
        point_count = optimized_state.positions_.shape[0]
        if self._inverse_hessian is None:
            self._inverse_hessian = self._calculate_inv_hessian(point_count)
        reshaped_grad = optimized_state.positions_.grad.reshape(point_count, -1)
        reshaped_grad = self._inverse_hessian @ reshaped_grad
        optimized_state.positions_.grad = reshaped_grad.reshape(optimized_state.positions_.grad.shape)

    def _calculate_inv_hessian(self, point_count):
        hessian = self._velocity_hessian_weight * self._calculate_velocity_hessian(point_count) + np.eye(point_count)
        inv_hessian = np.linalg.inv(hessian)
        return torch.tensor(inv_hessian.astype(np.float32), device=self._device)

    @staticmethod
    def _calculate_velocity_hessian(point_count):
        hessian = np.zeros((point_count, point_count), dtype=np.float32)
        for i in range(point_count):
            if i == 0:
                hessian[i, i] = 2
            elif i == point_count - 1:
                hessian[i, i] = 2
            else:
                hessian[i, i] = 4
            if i > 0:
                hessian[i, i - 1] = -2
                hessian[i - 1, i] = -2
        return hessian

    @staticmethod
    def _calculate_acceleration_hessian(point_count):
        hessian = np.zeros((point_count, point_count))

        for i in range(0, point_count):
            if i == 0:
                hessian[i, i] = 2
            elif i == 1:
                hessian[i, i] = 10
            elif i == point_count - 1:
                hessian[i, i] = 2
            elif i == point_count - 2:
                hessian[i, i] = 10
            else:
                hessian[i, i] = 12
            if i == 1:
                hessian[i, i - 1] = -4
                hessian[i - 1, i] = -4
            elif i == point_count - 1:
                hessian[i, i - 1] = -4
                hessian[i - 1, i] = -4
            else:
                hessian[i, i - 1] = -8
                hessian[i - 1, i] = -8
            if i > 2:
                hessian[i, i - 2] = 2
                hessian[i - 2, i] = 2

        return hessian


class PathOptimizer:
    def __init__(self, timer: Timer, optimizer: OptimizerWithLagrangeMultipliers,
                 loss_builder: MultiRobotPathLossBuilder,
                 state_initializer: PathOptimizedStateInitializer, grad_preconditioner: GradPreconditioner):
        self._loss_builder = loss_builder
        self._optimizer = optimizer
        self._timer = timer
        self._grad_preconditioner = grad_preconditioner
        self._state_initializer = state_initializer
        self._optimized_state: Optional[MultiRobotPathOptimizedState] = None

    def setup(self):
        self._optimized_state = self._state_initializer.init()
        self._optimizer.setup(self._optimized_state.optimized_parameters,
                              [self._optimized_state.direction_constraint_multipliers])

    def step(self, collision_model):
        self._optimizer.zero_grad()
        loss = self._loss_builder.get_loss(collision_model, self._optimized_state)
        self._timer.tick("trajectory_backward")
        loss.backward()
        self._timer.tock("trajectory_backward")
        self._timer.tick("inv_hes_grad_multiplication")
        self._grad_preconditioner.precondition(self._optimized_state)
        self._timer.tock("inv_hes_grad_multiplication")
        self._timer.tick("trajectory_optimizer_step")
        self._optimizer.step()
        self._timer.tock("trajectory_optimizer_step")

    def reparametrize(self):
        with torch.no_grad():
            self._optimized_state.reparametrize()

    @property
    def result_path(self) -> MultiRobotResultPath:
        return self._optimized_state.result_path


class CollisionModelPointSampler:
    def __init__(self, fine_random_offset: float, course_random_offset: float, angle_random_offset):
        self._fine_random_offset = fine_random_offset
        self._course_random_offset = course_random_offset
        self._angle_random_offset = angle_random_offset
        self._positions = None

    def sample(self, result_path: MultiRobotResultPath) -> np.ndarray:
        positions: np.ndarray = result_path.numpy_positions
        if self._positions is None:
            self._positions = np.zeros((0, positions.shape[1], positions.shape[2]))
        points = positions[:, :, :2]
        angles = positions[:, :, 2]

        course_points = points + np.random.randn(*points.shape) * self._course_random_offset
        fine_points = points + np.random.randn(*points.shape) * self._fine_random_offset
        points = np.concatenate([course_points, fine_points], axis=0)
        angles = np.concatenate([angles, angles], axis=0) + np.random.randn(2 * angles.shape[0],
                                                                            angles.shape[1]) * self._angle_random_offset

        positions = np.concatenate([points, angles[:, :, None]], axis=2)
        self._positions = np.concatenate([self._positions, positions], axis=0)
        if self._positions.shape[0] > 1000:
            self._positions = self._positions[-1000:]
        return positions


@dataclasses.dataclass
class ONFModelConfig:
    mean: float
    sigma: float
    input_dimension: int
    encoding_dimension: int
    output_dimension: int
    hidden_dimensions: List[int]


class ONF(nn.Module):
    def __init__(self, parameters: ONFModelConfig):
        super().__init__()
        self.encoding_layer = nn.Linear(parameters.input_dimension, parameters.encoding_dimension)
        self.mlp1 = self.make_mlp(parameters.encoding_dimension, parameters.hidden_dimensions[:-1],
                                  parameters.hidden_dimensions[-1])
        self.mlp2 = self.make_mlp(parameters.encoding_dimension + parameters.hidden_dimensions[-1], [],
                                  parameters.output_dimension)
        self._mean = parameters.mean
        self._sigma = parameters.sigma

    @classmethod
    def make_mlp(cls, input_dimension, hidden_dimensions, output_dimension):
        modules = []
        for dimension in hidden_dimensions:
            modules.append(nn.Linear(input_dimension, dimension))
            modules.append(nn.ReLU())
            input_dimension = dimension
        modules.append(nn.Linear(input_dimension, output_dimension))
        return nn.Sequential(*modules)

    def forward(self, x):
        x = (x - self._mean) / self._sigma
        x = self.encoding_layer(x)
        x = torch.sin(x)
        input_x = x
        x = self.mlp1(input_x)
        x = torch.cat([x, input_x], dim=1)
        x = self.mlp2(x)
        return x


class CollisionModelFactory:
    def __init__(self, parameters: ONFModelConfig):
        self._parameters = parameters

    def make_collision_model(self):
        return ONF(self._parameters)


class CollisionNeuralFieldModelTrainer:
    def __init__(self, timer: Timer, planner_task: MultiRobotPathPlannerTask,
                 collision_model_point_sampler: CollisionModelPointSampler,
                 optimizer: OptimizerImpl, collision_model_factory: CollisionModelFactory,
                 device: str):
        self._timer = timer
        self._collision_model_point_sampler = collision_model_point_sampler
        self._optimizer = optimizer
        self._collision_model_factory = collision_model_factory
        self._planner_task = planner_task
        self._collision_model: Optional[nn.Module] = None
        self._device = device
        self._collision_loss_function = nn.BCEWithLogitsLoss()
        self._timer = timer

    def setup(self):
        self._collision_model = self._collision_model_factory.make_collision_model()
        self._optimizer.setup(self._collision_model.parameters())

    def learning_step(self, path: MultiRobotResultPath):
        self._timer.tick("optimize_collision_model")
        points = self._collision_model_point_sampler.sample(path)
        self._collision_model.requires_grad_(True)
        self._optimizer.zero_grad()
        predicted_collision = self._calculate_predicted_collision(points)
        truth_collision = self._calculate_truth_collision(points)
        truth_collision = torch.tensor(truth_collision.astype(np.float32), device=self._device)
        loss = self._collision_loss_function(predicted_collision, truth_collision)
        loss.backward()
        self._optimizer.step()
        self._collision_model.requires_grad_(False)
        self._timer.tock("optimize_collision_model")

    def _calculate_predicted_collision(self, positions: np.array):
        positions = positions.reshape(positions.shape[0], -1)
        result = self._collision_model(torch.tensor(positions.astype(np.float32), device=self._device))
        return result

    def _calculate_truth_collision(self, positions: np.array):
        result = self._planner_task.collision_detector.is_collision_for_each_robot_for_list(
            [PositionArray2D.from_vec(x) for x in positions])
        return result

    @property
    def collision_model(self) -> nn.Module:
        return self._collision_model


class WarehouseNFOMP:
    def __init__(self, planner_task: MultiRobotPathPlannerTask,
                 collision_neural_field_model_trainer: CollisionNeuralFieldModelTrainer, path_optimizer: PathOptimizer,
                 iterations: int, reparametrize_rate: int):
        self._reparametrize_rate = reparametrize_rate
        self.planner_task = planner_task
        self._path_optimizer = path_optimizer
        self._collision_neural_field_model_trainer = collision_neural_field_model_trainer
        self._iterations = iterations
        self._current_iteration = 0

    def plan(self) -> MultiRobotResultPath:
        self.setup()
        for i in tqdm(range(self._iterations)):
            self.step()
        self._path_optimizer.reparametrize()
        return self.get_result()

    def setup(self):
        self._path_optimizer.setup()
        self._collision_neural_field_model_trainer.setup()
        self._current_iteration = 0

    def get_result(self) -> MultiRobotResultPath:
        return self._path_optimizer.result_path

    def step(self):
        path: MultiRobotResultPath = self._path_optimizer.result_path
        self._collision_neural_field_model_trainer.learning_step(path)
        collision_model = self._collision_neural_field_model_trainer.collision_model
        self._path_optimizer.step(collision_model)
        if self._reparametrize_rate != -1 and self._current_iteration % self._reparametrize_rate == 0:
            self._path_optimizer.reparametrize()
        self._current_iteration += 1
