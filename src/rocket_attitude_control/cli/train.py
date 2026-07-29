"""Train an A2C policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback

from rocket_attitude_control.env import RocketEnv


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--timesteps", type=int, default=100_000)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--device", default="auto")
    command.add_argument("--output", type=Path, default=Path("models/a2c_latest.zip"))
    command.add_argument("--checkpoint-freq", type=int, default=100_000)
    command.add_argument("--tensorboard-log", type=Path)
    return command


def main() -> None:
    args = parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = RocketEnv()
    model = A2C(
        "MultiInputPolicy",
        env,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=False,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(args.tensorboard_log) if args.tensorboard_log else None,
        verbose=1,
    )
    checkpoint = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(args.output.parent / "checkpoints"),
        name_prefix="a2c_rocket",
    )
    model.learn(total_timesteps=args.timesteps, callback=checkpoint)
    model.save(str(args.output.with_suffix("")))
    env.close()


if __name__ == "__main__":
    main()
