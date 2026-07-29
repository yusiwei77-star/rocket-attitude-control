"""Gymnasium adapter for the rocket simulation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .rendering import PygameRenderer
from .simulation import RocketSimulation, SimulationState


class RocketEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self.simulation = RocketSimulation()
        self.action_space = spaces.MultiBinary(8)
        self.observation_space = spaces.Dict(
            {
                "angle_error": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "angle_velocity_error": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
            }
        )
        self._renderer: PygameRenderer | None = None

    def _observation(self, state: SimulationState) -> dict[str, np.ndarray]:
        return {
            "angle_error": (state.angles - state.nominal_angles).astype(np.float32),
            "angle_velocity_error": (
                state.angular_rates - state.nominal_rates
            ).astype(np.float32),
        }

    @staticmethod
    def _info(state: SimulationState) -> dict[str, Any]:
        return {
            "t": np.float32(state.time),
            "angle": state.angles.astype(np.float32),
            "angle_velocity": state.angular_rates.astype(np.float32),
            "F": state.thrust.astype(np.float32),
            "nominal_attitude": state.nominal_angles.astype(np.float32),
            "nominal_angular_velocity": state.nominal_rates.astype(np.float32),
            "fuel": np.float32(state.fuel),
            "score": np.float32(state.score),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        state = self.simulation.reset(self.np_random)
        if self.render_mode == "human":
            self.render()
        return self._observation(state), self._info(state)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        state, reward, terminated = self.simulation.step(action)
        if self.render_mode == "human":
            self.render()
        return self._observation(state), reward, terminated, False, self._info(state)

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = PygameRenderer(display=self.render_mode == "human")
        frame = self._renderer.render(self.simulation.state)
        return frame if self.render_mode == "rgb_array" else None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
