import numpy as np

__all__ = ["delta_angle", "abs_delta_angle"]


def delta_angle(rad1, rad2):
    """Calculate the difference between two angles."""
    delta = rad2 - rad1
    if delta > np.pi:
        delta -= 2 * np.pi
    elif delta < -np.pi:
        delta += 2 * np.pi
    return delta


def abs_delta_angle(rad1, rad2):
    """Calculate the absolute value of difference between two angles."""
    return np.abs(delta_angle(rad1, rad2))
