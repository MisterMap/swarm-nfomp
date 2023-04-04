import dataclasses
from typing import Any, Optional, List

import numpy as np

from swarm_nfomp.collision_detector.point_array_collision_detector import PointArrayCollisionDetector
from swarm_nfomp.utils.point2d import Point2D
from swarm_nfomp.utils.point_array2d import PointArray2D


@dataclasses.dataclass
class RectangleBounds2D:
    max_x: float
    min_x: float
    max_y: float
    min_y: float


@dataclasses.dataclass
class Point2DPlannerTask:
    start: Point2D
    goal: Point2D
    collision_detector: PointArrayCollisionDetector
    bounds: RectangleBounds2D


@dataclasses.dataclass
class Point2DPath:
    points: np.ndarray

    @classmethod
    def from_list(cls, points: List[Point2D]):
        return cls(np.array([x.as_numpy() for x in points]))


@dataclasses.dataclass
class TreeNode:
    point: Point2D
    parent: Any
    index: int


class Tree:
    def __init__(self, node: TreeNode):
        self.root = node
        self.points = node.point.as_numpy()[None]
        self.nodes = [node]

    def add_point(self, point: Point2D, parent):
        node = TreeNode(point, parent, self.points.shape[0])
        self.points = np.vstack((self.points, point.as_numpy()))
        self.nodes.append(
            node)
        return node

    def nearest_node(self, point: Point2D) -> TreeNode:
        distances = np.linalg.norm(self.points - point.as_numpy(), axis=1)
        index = np.argmin(distances)
        return self.nodes[index]


@dataclasses.dataclass
class RRTStarParameters:
    collision_step: float
    iterations: int
    steer_distance: float
    goal_point_probability: float


def steer_2d(point1: Point2D, point2: Point2D, steer_distance: float) -> Point2D:
    distance = np.linalg.norm(point2.as_numpy() - point1.as_numpy())
    if distance < steer_distance:
        return point2
    direction = (point2.as_numpy() - point1.as_numpy()) / distance
    return Point2D(point1.x + steer_distance * direction[0], point1.y + steer_distance * direction[1])


def interpolate_points_2d(point1: Point2D, point2: Point2D, interpolation_count: int):
    x = np.linspace(point1.x, point2.x, interpolation_count)
    y = np.linspace(point1.y, point2.y, interpolation_count)
    return PointArray2D(x, y)


class RRTPoint2DPlanner:
    def __init__(self, parameters: RRTStarParameters):
        self.planner_task = None
        self.tree: Optional[Tree] = None
        self._parameters = parameters

    def set_planner_task(self, planner_task: Point2DPlannerTask):
        self.planner_task = planner_task
        self.tree = Tree(TreeNode(self.planner_task.start, None, 0))

    def plan(self) -> Point2DPath:
        for i in range(self._parameters.iterations):
            random_point = self._sample_random_point_with_goal_point()
            nearest_node = self.tree.nearest_node(random_point)
            new_point = steer_2d(nearest_node.point, random_point, self._parameters.steer_distance)
            if self._is_path_free(nearest_node.point, new_point):
                self.tree.add_point(new_point, nearest_node)
        return self._calculate_path()

    def _sample_random_point_with_goal_point(self):
        if np.random.uniform() < self._parameters.goal_point_probability:
            return self.planner_task.goal
        return self._sample_random_point()

    def _sample_random_point(self):
        bounds = self.planner_task.bounds
        return Point2D(np.random.uniform(bounds.min_x, bounds.max_x), np.random.uniform(bounds.min_y, bounds.max_y))

    def _find_nearest_point(self, point: Point2D) -> Point2D:
        return Point2D.from_vec(
            self.tree.points[np.argmin(np.linalg.norm(self.tree.points - point.as_numpy(), axis=1))])

    def _is_path_free(self, random_point: Point2D, new_point: Point2D):
        points_count = int(random_point.distance(new_point) / self._parameters.collision_step) + 1
        points = interpolate_points_2d(random_point, new_point, points_count)
        return not np.any(self.planner_task.collision_detector.is_collision_for_array(points))

    def _calculate_path(self):
        node = self.tree.nearest_node(self.planner_task.goal)
        path = [node.point]
        index = node.index
        while index != 0:
            node = node.parent
            path.append(node.point)
            index = node.index
        return Point2DPath.from_list(path[::-1])
