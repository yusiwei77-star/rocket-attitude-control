"""Generate synchronized plot and rocket-interface MP4 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rocket_attitude_control.rollout import (
    load_a2c,
    run_episode,
    trajectory_metrics,
    write_json,
)
from rocket_attitude_control.video import (
    PREVIEW_FPS,
    render_gif_preview,
    render_videos,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--model", type=Path, default=Path("models/a2c_2460000.zip"))
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--device", default="auto")
    command.add_argument("--output-dir", type=Path, default=Path("artifacts/demos"))
    command.add_argument("--frames", type=int, default=327)
    command.add_argument("--fps", type=int, default=30)
    command.add_argument("--preview-fps", type=int, default=PREVIEW_FPS)
    command.add_argument("--no-previews", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    trajectory = run_episode(load_a2c(args.model, args.device), args.seed)
    prefix = f"a2c_2460000_seed{args.seed}"
    plot_path, rocket_path = render_videos(
        trajectory,
        args.output_dir,
        frames=args.frames,
        fps=args.fps,
        prefix=prefix,
    )
    preview_path = None
    if not args.no_previews:
        preview_path = render_gif_preview(
            plot_path,
            rocket_path,
            fps=args.preview_fps,
        )
    result_dir = args.output_dir.parent / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    trajectory.save(result_dir / f"{prefix}_trajectory.npz")
    metrics = trajectory_metrics(trajectory, args.seed)
    metrics.update(
        {
            "weight": str(args.model),
            "algorithm": "A2C",
            "video_frames": args.frames,
            "video_fps": args.fps,
            "video_duration_s": args.frames / args.fps,
        }
    )
    write_json(metrics, result_dir / f"{prefix}_summary.json")
    outputs = {"plot_video": str(plot_path), "rocket_video": str(rocket_path)}
    if preview_path is not None:
        outputs["synchronized_preview"] = str(preview_path)
    print(json.dumps({**outputs, **metrics}, indent=2))


if __name__ == "__main__":
    main()
