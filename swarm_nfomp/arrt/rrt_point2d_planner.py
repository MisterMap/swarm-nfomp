import dataclasses

import numpy as np

from swarm_nfomp.arrt.rrt_planner import RRTPlanner
from swarm_nfomp.collision_detector.point_array_collision_detector import PointArrayCollisionDetector
from swarm_nfomp.planner.planner import Bounds
from swarm_nfomp.utils.point2d import Point2D


@dataclasses.dataclass
class RectangleBounds2D(Bounds[Point2D]):
    max_x: float
    min_x: float
    max_y: float
    min_y: float

    def sample_random_point(self) -> Point2D:
        return Point2D(
            np.random.uniform(self.min_x, self.max_x),
            np.random.uniform(self.min_y, self.max_y)
        )


@dataclasses.dataclass
class Point2DPlannerTask:
    start: Point2D
    goal: Point2D
    collision_detector: PointArrayCollisionDetector
    bounds: RectangleBounds2D


class RRTPoint2DPlanner(RRTPlanner[Point2D]):
    pass
