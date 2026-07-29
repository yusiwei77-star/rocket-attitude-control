"""Policy evaluation and trajectory collection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .env import RocketEnv


@dataclass
class Trajectory:
    time: NDArray[np.float64]
    angles_deg: NDArray[np.float64]
    rates_deg_s: NDArray[np.float64]
    nominal_angles_deg: NDArray[np.float64]
    nominal_rates_deg_s: NDArray[np.float64]
    thrust_n: NDArray[np.float64]
    moments_nm: NDArray[np.float64]
    fuel_ns: NDArray[np.float64]
    rewards: NDArray[np.float64]
    score: float

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **asdict(self))


def load_a2c(path: str | Path, device: str = "auto"):
    from stable_baselines3 import A2C

    return A2C.load(str(path), device=device)


def run_episode(model, seed: int = 0) -> Trajectory:
    env = RocketEnv()
    observation, _ = env.reset(seed=seed)
    rows: dict[str, list] = {
        "time": [],
        "angles": [],
        "rates": [],
        "nominal_angles": [],
        "nominal_rates": [],
        "thrust": [],
        "moments": [],
        "fuel": [],
        "rewards": [],
    }
    terminated = truncated = False
    score = 0.0
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        state = env.simulation.state
        rows["time"].append(state.time)
        rows["angles"].append(np.rad2deg(state.angles))
        rows["rates"].append(np.rad2deg(state.angular_rates))
        rows["nominal_angles"].append(np.rad2deg(state.nominal_angles))
        rows["nominal_rates"].append(np.rad2deg(state.nominal_rates))
        rows["thrust"].append(state.thrust)
        rows["moments"].append(state.moments)
        rows["fuel"].append(state.fuel)
        rows["rewards"].append(reward)
        score = float(info["score"])
    env.close()
    return Trajectory(
        time=np.asarray(rows["time"]),
        angles_deg=np.asarray(rows["angles"]),
        rates_deg_s=np.asarray(rows["rates"]),
        nominal_angles_deg=np.asarray(rows["nominal_angles"]),
        nominal_rates_deg_s=np.asarray(rows["nominal_rates"]),
        thrust_n=np.asarray(rows["thrust"]),
        moments_nm=np.asarray(rows["moments"]),
        fuel_ns=np.asarray(rows["fuel"]),
        rewards=np.asarray(rows["rewards"]),
        score=score,
    )


def trajectory_metrics(trajectory: Trajectory, seed: int) -> dict[str, object]:
    index_70 = int(np.argmin(np.abs(trajectory.time - 70.0)))
    final_angles = trajectory.angles_deg[-1]
    final_rates = trajectory.rates_deg_s[-1]
    pass_70 = bool(
        np.all(np.abs(trajectory.rates_deg_s[index_70]) < np.array([0.5, 1.0, 1.0]))
    )
    pass_130 = bool(
        np.all(np.abs(final_angles - np.array([0.0, 0.0, 120.0])) < 3.0)
        and np.all(np.abs(final_rates) < np.array([0.5, 1.0, 1.0]))
    )
    return {
        "seed": seed,
        "score": trajectory.score,
        "fuel_Ns": float(trajectory.fuel_ns[-1]),
        "angle_rmse_deg": float(
            np.sqrt(np.mean((trajectory.angles_deg - trajectory.nominal_angles_deg) ** 2))
        ),
        "rate_rmse_deg_s": float(
            np.sqrt(np.mean((trajectory.rates_deg_s - trajectory.nominal_rates_deg_s) ** 2))
        ),
        "pass_70s_rate_constraint": pass_70,
        "pass_130s_final_constraint": pass_130,
        "final_angles_deg": final_angles.tolist(),
        "final_rates_deg_s": final_rates.tolist(),
    }


def evaluate(model, episodes: int = 20, start_seed: int = 0) -> dict[str, object]:
    runs = [
        trajectory_metrics(run_episode(model, seed), seed)
        for seed in range(start_seed, start_seed + episodes)
    ]
    return {
        "episodes": episodes,
        "start_seed": start_seed,
        "success_70_pct": 100.0
        * float(np.mean([run["pass_70s_rate_constraint"] for run in runs])),
        "success_130_pct": 100.0
        * float(np.mean([run["pass_130s_final_constraint"] for run in runs])),
        "fuel_mean_Ns": float(np.mean([run["fuel_Ns"] for run in runs])),
        "fuel_std_Ns": float(np.std([run["fuel_Ns"] for run in runs])),
        "score_mean": float(np.mean([run["score"] for run in runs])),
        "angle_rmse_mean_deg": float(np.mean([run["angle_rmse_deg"] for run in runs])),
        "rate_rmse_mean_deg_s": float(
            np.mean([run["rate_rmse_deg_s"] for run in runs])
        ),
        "runs": runs,
    }


def write_json(data: dict[str, object], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
