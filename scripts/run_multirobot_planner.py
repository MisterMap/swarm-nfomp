import copy
import os
from multiprocessing import Queue, Process

import clearml
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

VISUALIZATION_FILE_NAME = "data/warehouse_nfomp.png"


def log_figure(logger: clearml.Logger, iteration: int):
    logger.report_matplotlib_figure("warehouse_nfomp_title", "warehouse_nfomp_series", plt.gcf(),
                                    report_image=False, report_interactive=False,
                                    iteration=iteration)


def plot_process_function(queue: Queue, logger: clearml.Logger):
    plotter = WarehouseNfompMatplotlibPlotter()
    previous_log_iteration = -1
    while True:
        queue_size = queue.qsize()
        previous_queue_result = None
        for i in range(queue_size - 1):
            previous_queue_result = queue.get()
        queue_result = queue.get()

        if queue_result is not None:
            planner_task, result = queue_result
            plotter.show(planner_task, result)
            if result.iteration - previous_log_iteration >= 100:
                log_figure(logger, result.iteration)
                previous_log_iteration = result.iteration
        elif previous_queue_result is not None:
            planner_task, result = previous_queue_result
            plotter.show(planner_task, result)
            log_figure(logger, result.iteration)
            break
        else:
            break
    plotter.save("data/warehouse_nfomp.png")


def load_config(path):
    with open(path, "r") as f:
        parameters = yaml.safe_load(f)
    return parameters


def main():
    task: clearml.Task = clearml.Task.init(project_name="warehouse-nfomp", task_name="run_multirobot_planner",
                                           task_type=clearml.Task.TaskTypes.inference, auto_connect_frameworks=False,
                                           reuse_last_task_id=True)
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_config_path = os.path.join(parent_path, "configs/multi_robot_planner_tasks/four_robot_task_random.yaml")
    planner_config_path = os.path.join(parent_path, "configs/nfomp_planners/warehouse_nfomp.yaml")

    task_config = load_config(task_config_path)
    task.connect(task_config, name="task_config")
    planner_config = load_config(planner_config_path)
    task.connect(planner_config, name="planner_config")

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
    process = Process(target=plot_process_function, args=(queue, task.get_logger()))
    process.start()
    result = None
    for i in tqdm(range(iterations)):
        planner.step()
        result = planner.get_result()
        result.iteration = i
        queue.put((copy.deepcopy(planner.planner_task), copy.deepcopy(result)))
    queue.put(None)
    process.join()
    visualizer_parameters = CollisionDetectionResultVisualizerConfig(
        xmin=-10, xmax=10, ymin=-5, ymax=5)
    visualizer = WarehousePathPlannerResultVisualizer(parameters=visualizer_parameters)
    visualizer.visualize(planner_task.collision_detector, result)
    visualizer.save("data/warehouse_path_planner_result.html")
    task.upload_artifact("warehouse_path_planner_result", "data/warehouse_path_planner_result.html")
    task.close()


if __name__ == '__main__':
    main()
