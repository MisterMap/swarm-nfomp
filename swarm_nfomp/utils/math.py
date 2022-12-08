import numpy as np


def wrap_angles(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi
