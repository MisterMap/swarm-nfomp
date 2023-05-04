import copy
import os
from multiprocessing import Queue, Process

import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from swarm_nfomp.utils.timer import Timer
from swarm_nfomp.utils.universal_factory import UniversalFactory
from swarm_nfomp.warehouse_nfomp.warehouse_nfomp import MultiRobotPathPlannerTask, WarehouseNFOMP
from swarm_nfomp.warehouse_nfomp.warehouse_nfomp_matplotlib_plotter import WarehouseNfompMatplotlibPlotter
from swarm_nfomp.warehouse_nfomp.warehouse_nfomp_visualizer import CollisionDetectionResultVisualizerConfig, \
    WarehousePathPlannerResultVisualizer


def plot_process_function(queue: Queue):
    plotter = WarehouseNfompMatplotlibPlotter()
    while True:
        queue_size = queue.qsize()
        previous_queue_result = None
        for i in range(queue_size - 1):
            previous_queue_result = queue.get()
        queue_result = queue.get()

        if queue_result is not None:
            planner_task, result = queue_result
            plotter.show(planner_task, result)
        elif previous_queue_result is not None:
            planner_task, result = previous_queue_result
            plotter.show(planner_task, result)
            break
        else:
            break
    plotter.save("data/warehouse_nfomp.png")


def load_config(path):
    with open(path, "r") as f:
        parameters = yaml.safe_load(f)
    return parameters


def main():
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_config_path = os.path.join(parent_path, "configs/multi_robot_planner_tasks/four_robot_task_random.yaml")
    planner_config_path = os.path.join(parent_path, "configs/nfomp_planners/warehouse_nfomp.yaml")

    task_config = load_config(task_config_path)
    planner_config = load_config(planner_config_path)

    factory = UniversalFactory(MultiRobotPathPlannerTask, WarehouseNFOMP)

    torch.manual_seed(100)
    iterations = 1500
    global_timer = Timer()
    device_parameter = "cpu"
    planner_task = factory.make(parameters=task_config)
    robot_count = len(planner_task.collision_detector.robot_shapes)
    planner = factory.make(parameters=planner_config, planner_task=planner_task, timer=global_timer,
                           device=device_parameter, input_dimension=3 * robot_count, output_dimension=robot_count,
                           iterations=iterations)
    planner.setup()
    queue = Queue()
    process = Process(target=plot_process_function, args=(queue,))
    process.start()
    result = None
    for i in tqdm(range(iterations)):
        planner.step()
        result = planner.get_result()
        queue.put((copy.deepcopy(planner.planner_task), copy.deepcopy(result)))
    queue.put(None)
    process.join()
    visualizer_parameters = CollisionDetectionResultVisualizerConfig(
        xmin=-10, xmax=10, ymin=-5, ymax=5)
    visualizer = WarehousePathPlannerResultVisualizer(parameters=visualizer_parameters)
    visualizer.visualize(planner_task.collision_detector, result)
    visualizer.save("data/warehouse_path_planner_result.html")


if __name__ == '__main__':
    main()
