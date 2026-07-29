"""Headless rigid-body rocket attitude simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .nominal import NominalAttitudeController


DT = 0.1
END_TIME = 130.0
TRANSITION_TIME = 70.0
THRUST = 400.0


@dataclass(frozen=True)
class SimulationState:
    time: float
    angles: NDArray[np.float64]
    angular_rates: NDArray[np.float64]
    nominal_angles: NDArray[np.float64]
    nominal_rates: NDArray[np.float64]
    thrust: NDArray[np.float64]
    moments: NDArray[np.float64]
    fuel: float
    score: float


class RocketSimulation:
    """Physics and reward model without any display or wall-clock delay."""

    inertia = np.array([1.4e5, 1.2e7, 1.2e7], dtype=np.float64)

    def __init__(self, dt: float = DT, end_time: float = END_TIME) -> None:
        self.dt = float(dt)
        self.end_time = float(end_time)
        self.max_steps = round(self.end_time / self.dt)
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> SimulationState:
        rng = rng if rng is not None else np.random.default_rng()
        self.step_count = 0
        self.omega = np.deg2rad(rng.uniform(-1.5, 1.5, size=3))
        self.angles = np.array([0.0, 0.0, np.deg2rad(35.0)])
        self.angles += np.deg2rad(rng.uniform(-5.0, 5.0, size=3))
        self.angular_rates = np.zeros(3, dtype=np.float64)
        self.thrust = np.zeros(8, dtype=np.float64)
        self.moments = np.zeros(3, dtype=np.float64)
        self.fuel = 0.0
        self.score = 0.0

        controller = NominalAttitudeController(
            self.angles,
            self.omega,
            np.deg2rad(np.array([0.0, 0.0, 120.0])),
            np.deg2rad(np.array([0.0, 0.0, 85.0 / 130.0])),
            TRANSITION_TIME,
            self.end_time,
        )
        self.nominal_times, self.nominal_angles, self.nominal_rates = controller.integrate(
            self.dt
        )
        self.nominal_index = 0
        return self.state

    @property
    def time(self) -> float:
        return self.step_count * self.dt

    @property
    def state(self) -> SimulationState:
        return SimulationState(
            time=self.time,
            angles=self.angles.copy(),
            angular_rates=self.angular_rates.copy(),
            nominal_angles=self.nominal_angles[self.nominal_index].copy(),
            nominal_rates=self.nominal_rates[self.nominal_index].copy(),
            thrust=self.thrust.copy(),
            moments=self.moments.copy(),
            fuel=self.fuel,
            score=self.score,
        )

    def _set_action(self, action: NDArray[np.int8]) -> None:
        values = np.asarray(action, dtype=np.int8)
        if values.shape != (8,):
            raise ValueError(f"Expected an 8-element action, got {values.shape}")
        if not np.isin(values, (0, 1)).all():
            raise ValueError("Each thruster action must be 0 or 1")
        self.thrust = values.astype(np.float64) * THRUST

    def _update_dynamics(self) -> None:
        f = self.thrust
        radius = 4.3 / 2.0
        length = 25.0
        theta = np.deg2rad(22.5)
        self.moments = np.array(
            [
                (f[0] + f[4] - f[3] - f[7]) * radius * np.sin(theta)
                + (f[2] + f[6] - f[1] - f[5]) * radius * np.cos(theta),
                (f[5] + f[6] - f[1] - f[2]) * length,
                (f[0] + f[7] - f[3] - f[4]) * length,
            ]
        )
        jx, jy, jz = self.inertia
        wx, wy, wz = self.omega
        omega_acceleration = np.array(
            [
                ((jy - jz) / jx) * wy * wz + self.moments[0] / jx,
                ((jz - jx) / jy) * wz * wx + self.moments[1] / jy,
                ((jx - jy) / jz) * wx * wy + self.moments[2] / jz,
            ]
        )
        self.omega += omega_acceleration * self.dt

        gamma, psi, _ = self.angles
        wx, wy, wz = self.omega
        self.angular_rates = np.array(
            [
                wx + np.sin(gamma) * np.tan(psi) * wy + np.cos(gamma) * np.tan(psi) * wz,
                np.cos(gamma) * wy - np.sin(gamma) * wz,
                np.sin(gamma) / np.cos(psi) * wy + np.cos(gamma) / np.cos(psi) * wz,
            ]
        )
        self.angles += self.angular_rates * self.dt

    def _reward(self) -> float:
        nominal_angle = self.nominal_angles[self.nominal_index]
        nominal_rate = self.nominal_rates[self.nominal_index]
        angle_error_deg = np.abs(np.rad2deg(self.angles - nominal_angle))
        reward = (1.0 - 0.1 * angle_error_deg.sum()) * self.dt
        reward += float((angle_error_deg < 3.0).sum()) * self.dt / 3.0
        converging = (self.angles - nominal_angle) * (
            self.angular_rates - nominal_rate
        ) <= 0
        reward += float(converging.sum()) * 0.3 * self.dt / 3.0

        rates_deg = np.abs(np.rad2deg(self.angular_rates))
        if self.step_count == round(TRANSITION_TIME / self.dt) and np.all(
            rates_deg < np.array([0.5, 1.0, 1.0])
        ):
            reward += 40.0
        if self.step_count == self.max_steps:
            angles_deg = np.rad2deg(self.angles)
            if np.all(rates_deg < np.array([0.5, 1.0, 1.0])) and np.all(
                np.abs(angles_deg - np.array([0.0, 0.0, 120.0])) < 3.0
            ):
                reward += 60.0
        reward -= self.thrust.sum() / THRUST / 20.0 * self.dt
        return float(reward)

    def step(self, action: NDArray[np.int8]) -> tuple[SimulationState, float, bool]:
        if self.step_count >= self.max_steps:
            raise RuntimeError("Episode is complete; call reset() before stepping again")
        self._set_action(action)
        self.nominal_index = self.step_count
        self._update_dynamics()
        self.step_count += 1
        reward = self._reward()
        self.fuel += float(self.thrust.sum() * self.dt)
        self.score += reward
        return self.state, reward, self.step_count >= self.max_steps
