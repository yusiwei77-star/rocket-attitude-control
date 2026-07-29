import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from rocket_attitude_control.env import RocketEnv
from rocket_attitude_control.nominal import NominalAttitudeController


def test_environment_contract() -> None:
    env = RocketEnv()
    check_env(env, warn=True)
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert info["t"] == 0.0
    observation, _, terminated, truncated, _ = env.step(np.zeros(8, dtype=np.int8))
    assert env.observation_space.contains(observation)
    assert not terminated
    assert not truncated
    env.close()


def test_reset_is_deterministic() -> None:
    env = RocketEnv()
    first, first_info = env.reset(seed=123)
    second, second_info = env.reset(seed=123)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(first_info["angle"], second_info["angle"])
    np.testing.assert_array_equal(
        first_info["angle_velocity"], second_info["angle_velocity"]
    )
    env.close()


def test_episode_ends_at_exactly_130_seconds() -> None:
    env = RocketEnv()
    env.reset(seed=0)
    terminated = False
    for step in range(1, 1_301):
        _, _, terminated, truncated, info = env.step(np.zeros(8, dtype=np.int8))
        assert not truncated
        assert terminated is (step == 1_300)
    assert info["t"] == pytest.approx(130.0)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(8, dtype=np.int8))
    env.close()


def test_nominal_trajectory_includes_both_endpoints() -> None:
    controller = NominalAttitudeController(
        np.zeros(3), np.zeros(3), np.ones(3), np.zeros(3), 70.0, 130.0
    )
    times, attitudes, rates = controller.integrate(0.1)
    assert len(times) == 1_301
    assert attitudes.shape == rates.shape == (1_301, 3)
    assert times[0] == 0.0
    assert times[-1] == 130.0
