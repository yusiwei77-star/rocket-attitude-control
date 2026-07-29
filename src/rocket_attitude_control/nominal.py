"""Nominal attitude trajectory generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class NominalAttitudeController:
    """Generate the two-phase nominal trajectory used by the trained policy."""

    def __init__(
        self,
        start_attitude: NDArray[np.float64],
        start_attitude_rate: NDArray[np.float64],
        end_attitude: NDArray[np.float64],
        end_attitude_rate: NDArray[np.float64],
        transition_time: float,
        end_time: float,
    ) -> None:
        self.start_attitude = np.asarray(start_attitude, dtype=np.float64)
        self.start_attitude_rate = np.asarray(start_attitude_rate, dtype=np.float64)
        self.end_attitude = np.asarray(end_attitude, dtype=np.float64)
        self.end_attitude_rate = np.asarray(end_attitude_rate, dtype=np.float64)
        self.transition_time = transition_time
        self.end_time = end_time

    def coefficients(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        t = self.transition_time
        matrix = np.array([[t, 0.5 * t**2], [0.5 * t**2, t**3 / 3.0]])
        c0 = np.zeros(3)
        c1 = np.zeros(3)
        for axis in range(3):
            target = np.array(
                [
                    self.end_attitude_rate[axis] - self.start_attitude_rate[axis],
                    (self.end_attitude[axis] - self.start_attitude[axis])
                    * self.transition_time
                    / self.end_time
                    - self.start_attitude_rate[axis] * self.transition_time,
                ]
            )
            c0[axis], c1[axis] = np.linalg.solve(matrix, target)
        return c0, c1

    def integrate(
        self, dt: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return time, attitude, and rate arrays, including both endpoints."""
        steps = round(self.end_time / dt)
        times = np.arange(steps + 1, dtype=np.float64) * dt
        attitudes = np.zeros((steps + 1, 3), dtype=np.float64)
        rates = np.zeros((steps + 1, 3), dtype=np.float64)
        attitudes[0] = self.start_attitude
        rates[0] = self.start_attitude_rate
        c0, c1 = self.coefficients()

        for index in range(1, steps + 1):
            previous_time = times[index - 1]
            acceleration = c0 + c1 * (self.transition_time - previous_time)
            if times[index] < self.transition_time:
                rates[index] = rates[index - 1] + acceleration * dt
            else:
                rates[index] = (
                    self.end_attitude - self.start_attitude
                ) / self.end_time
            attitudes[index] = (
                attitudes[index - 1]
                + rates[index - 1] * dt
                + 0.5 * acceleration * dt**2
            )
        return times, attitudes, rates
