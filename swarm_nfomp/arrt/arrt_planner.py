import dataclasses
from typing import Tuple, Optional

import numpy as np

from swarm_nfomp.arrt.a_star import AStar
from swarm_nfomp.arrt.rrt_planner import RRTParameters, RRTPlanner
from swarm_nfomp.planner.planner import State, Path


@dataclasses.dataclass
class ARRTPoint2DPlannerParameters(RRTParameters):
    a_star_iterations: int


@dataclasses.dataclass
class Plane:
    start_point: np.ndarray
    plane_vector_x: np.ndarray
    plane_vector_y: np.ndarray
    size_x: int
    size_y: int

    def get_vector(self, plane_point: tuple[float, float]) -> np.ndarray:
        return self.start_point + self.plane_vector_x * plane_point[0] + self.plane_vector_y * plane_point[1]


class ARRTPlanner(RRTPlanner[State]):
    def __init__(self, parameters: ARRTPoint2DPlannerParameters):
        super().__init__(parameters)
        self._parameters = parameters
        self._random_target_point = None
        self._random_plane_point = None
        self._nearest_point = None
        self._plane: Optional[Plane] = None

    def plan(self):
        for i in range(self._parameters.iterations):
            self.step()
        return self._calculate_path()

    def step(self):
        self._random_target_point = self._sample_random_point_with_goal_point()
        self._random_plane_point = self._sample_random_point()
        nearest_node = self.tree.nearest_node(self._random_target_point)
        self._nearest_point = nearest_node.point
        self._calculate_plane()
        node_dict = {(0, 0): nearest_node}
        a_star_planner = AStar((0, 0), (self._plane.size_x, 0), self._a_star_collision_function)
        a_star_planner.step()
        for i in range(self._parameters.a_star_iterations):
            result = a_star_planner.step()
            if result is None:
                break
            current_state, parent_state = result
            parent_node = node_dict[parent_state]
            node = self.tree.add_point(self._state_from_location(current_state), parent_node)
            node_dict[current_state] = node

    def _calculate_plane(self):
        delta_x = self._random_target_point - self._nearest_point
        delta_y = self._random_plane_point - self._nearest_point
        projection = delta_x * np.sum(delta_y * delta_x) / (np.linalg.norm(delta_x) ** 2 + 1e-6)
        delta_y = delta_y - projection
        size_x = int(np.linalg.norm(delta_x) / self._parameters.steer_distance) + 1
        size_y = int(np.linalg.norm(delta_y) / self._parameters.steer_distance) + 1
        self._plane = Plane(
            start_point=self._nearest_point.as_numpy(),
            size_x=size_x,
            size_y=size_y,
            plane_vector_x=delta_x / size_x,
            plane_vector_y=delta_y / size_y
        )

    def _a_star_collision_function(self, point_location1: Tuple[int, int], point_location2: Tuple[int, int]):
        point1 = self._state_from_location(point_location1)
        point2 = self._state_from_location(point_location2)
        return not self.planner_task.collision_detector.is_collision_between(point1, point2)

    def _state_from_location(self, current_state):
        vector = self._plane.get_vector(current_state)
        return self._state_from_vector(vector)

    def _state_from_vector(self, vector) -> State:
        raise NotImplementedError()

    def _path_from_list(self, array: list) -> Path[State]:
        raise NotImplementedError()
