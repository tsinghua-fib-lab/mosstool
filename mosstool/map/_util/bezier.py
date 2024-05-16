"""Bezier, a module for creating Bezier curves.
Version 1.1, from < BezierCurveFunction-v1.ipynb > on 2019-05-02
"""

from typing import List, Sequence, Union

import numpy as np

__all__ = ["Bezier"]


class Bezier:
    """
    Bezier, a module for creating Bezier curves.
    """

    @staticmethod
    def TwoPoints(t: float, P1: np.ndarray, P2: np.ndarray):
        """
        Returns a point between P1 and P2, parametised by t.

        Args:
        - t (float/int): a parameterisation.
        - P1 (numpy array): a point.
        - P2 (numpy array): a point.

        Returns:
        - Q1 (numpy array): a point.
        """

        if not isinstance(P1, np.ndarray) or not isinstance(P2, np.ndarray):
            raise TypeError("Points must be an instance of the numpy.ndarray!")
        if not isinstance(t, (int, float)):
            raise TypeError("Parameter t must be an int or float!")

        Q1 = (1 - t) * P1 + t * P2
        return Q1

    @staticmethod
    def Points(t: float, points: List[np.ndarray]) -> List[np.ndarray]:
        """
        Returns a list of points interpolated by the Bezier process

        Args:
        - t (float/int): a parameterisation.
        - points (numpy array): list of numpy arrays; points.

        Returns:
        - new_points: list of numpy arrays; points.
        """
        new_points = []
        for i1 in range(0, len(points) - 1):
            new_points += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
        return new_points

    @staticmethod
    def Point(t: float, points: List[np.ndarray]):
        """
        Returns a point interpolated by the Bezier process

        Args:
        - t (float/int): a parameterisation.
        - points (numpy array): list of numpy arrays; points.

        Returns:
        - newpoint: numpy array; a point.
        """
        new_points = points
        while len(new_points) > 1:
            new_points = Bezier.Points(t, new_points)
        return new_points[0]

    @staticmethod
    def Curve(
        t_values: Union[Sequence[float], np.ndarray], points: List[np.ndarray]
    ) -> np.ndarray:
        """
        Returns a point interpolated by the Bezier process

        Args:
        - t_values: list of floats/ints; a parameterisation.
        - points: list of numpy arrays; points.

        Returns:
        - curve: list of numpy arrays; points.
        """

        if not hasattr(t_values, "__iter__"):
            raise TypeError(
                "`t_values` Must be an iterable of integers or floats, of length greater than 0 ."
            )
        if len(t_values) < 1:
            raise TypeError(
                "`t_values` Must be an iterable of integers or floats, of length greater than 0 ."
            )
        if not isinstance(t_values[0], (int, float)):
            raise TypeError(
                "`t_values` Must be an iterable of integers or floats, of length greater than 0 ."
            )

        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)
        curve = np.delete(curve, 0, 0)
        return curve


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    strength = 50
    p0 = np.array([0, 0])
    p1 = np.array([0, strength])
    p2 = np.array([100 - strength, 100])
    p3 = np.array([100, 100])
    t = np.linspace(0, 1, 100)
    curve = Bezier.Curve(t, [p0, p1, p2, p3])
    plt.plot(curve[:, 0], curve[:, 1])
    # p0 -> p1
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]])
    # p2 -> p3
    plt.plot([p2[0], p3[0]], [p2[1], p3[1]])
    plt.show()
