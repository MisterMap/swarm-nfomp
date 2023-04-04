from matplotlib import pyplot as plt

from swarm_nfomp.arrt.rrt_point2d_planner import Tree, Point2DPath
from swarm_nfomp.collision_detector.point_array_collision_detector import PointArrayCollisionDetector


def show_tree(tree: Tree, fig):
    ax = fig.gca()

    plt.scatter(tree.points[:, 0], tree.points[:, 1], color='black', s=1)

    for node in tree.nodes:
        if node.parent is not None:
            ax.plot([node.point.x, node.parent.point.x], [node.point.y, node.parent.point.y], color='blue')


def show_array_collision_detector(collision_detector: PointArrayCollisionDetector, fig):
    ax = fig.gca()
    for region in collision_detector.inside_rectangle_region_array:
        ax.add_patch(
            plt.Rectangle(
                (region[0], region[2]),  # (x,y)
                region[1] - region[0],  # width
                region[3] - region[2],  # height
                color='black'
            )
        )


def show_point(start_point, fig, color="green"):
    ax = fig.gca()
    ax.scatter([start_point.x], [start_point.y], color=color, s=20)


def show_path(path: Point2DPath, fig):
    plt.plot(path.points[:, 0], path.points[:, 1], color='red')
