from pathlib import Path

import pytest

from rocket_attitude_control.rollout import evaluate, load_a2c, run_episode


MODEL = Path("models/a2c_2460000.zip")


def test_checkpoint_loads_and_completes_episode() -> None:
    trajectory = run_episode(load_a2c(MODEL, device="cpu"), seed=0)
    assert len(trajectory.time) == 1_300
    assert trajectory.time[-1] == pytest.approx(130.0)


@pytest.mark.regression
def test_twenty_seed_policy_regression() -> None:
    result = evaluate(load_a2c(MODEL, device="cpu"), episodes=20, start_seed=0)
    assert result["success_130_pct"] == 100.0
    assert result["fuel_mean_Ns"] <= 30_000
    assert result["angle_rmse_mean_deg"] <= 1.0
