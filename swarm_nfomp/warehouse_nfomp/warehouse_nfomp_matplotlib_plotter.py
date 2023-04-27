import os

from matplotlib import pyplot as plt

from swarm_nfomp.collision_detector.multi_robot_collision_detector import MultiRobotCollisionDetector
from swarm_nfomp.plotting.plotting import show_multipolygon, show_transformed_robot_shapes, show_position_array2d
from swarm_nfomp.warehouse_nfomp.warehouse_nfomp import MultiRobotResultPath, MultiRobotPathPlannerTask


class WarehouseNfompMatplotlibPlotter:
    def __init__(self):
        self._fig = plt.figure(dpi=200)

    def show(self, planner_task: MultiRobotPathPlannerTask, result: MultiRobotResultPath):
        self._fig.clear()
        ax = self._fig.gca()
        ax.set_aspect('equal', adjustable='box')
        bounds = planner_task.bounds
        ax.set_xlim(bounds.min_x, bounds.max_x)
        ax.set_ylim(bounds.min_y, bounds.max_y)
        self.show_robot_collision_detector(planner_task.collision_detector)
        self.show_robot_paths(planner_task.collision_detector.robot_shapes, result)
        plt.pause(0.01)

    def save(self, filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._fig.savefig(filename)

    def show_robot_collision_detector(self, collision_detector: MultiRobotCollisionDetector):
        show_multipolygon(collision_detector.inside_rectangle_region, self._fig, color="black")

    def show_robot_paths(self, robot_shapes, result: MultiRobotResultPath):
        for robot_shape, path in zip(robot_shapes, result.robot_paths):
            show_position_array2d(path, self._fig, color="red")
