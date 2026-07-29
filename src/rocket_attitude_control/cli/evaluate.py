"""Evaluate an A2C checkpoint across deterministic seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rocket_attitude_control.rollout import evaluate, load_a2c, write_json


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--model", type=Path, default=Path("models/a2c_2460000.zip"))
    command.add_argument("--episodes", type=int, default=20)
    command.add_argument("--start-seed", type=int, default=0)
    command.add_argument("--device", default="auto")
    command.add_argument("--output", type=Path, default=Path("artifacts/results/evaluation.json"))
    return command


def main() -> None:
    args = parser().parse_args()
    result = evaluate(load_a2c(args.model, args.device), args.episodes, args.start_seed)
    write_json(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
