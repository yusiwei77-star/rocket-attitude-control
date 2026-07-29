"""Synchronized MP4 generation for a collected trajectory."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib
import numpy as np

from .rendering import PygameRenderer
from .rollout import Trajectory
from .simulation import SimulationState

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PLOT_VIDEO_SIZE = (1800, 1120)
ROCKET_VIDEO_SIZE = (PygameRenderer.width, PygameRenderer.height)
PREVIEW_HEIGHT = 588
PLOT_PREVIEW_SIZE = (
    round(PLOT_VIDEO_SIZE[0] * PREVIEW_HEIGHT / PLOT_VIDEO_SIZE[1]),
    PREVIEW_HEIGHT,
)
ROCKET_PREVIEW_SIZE = (
    round(ROCKET_VIDEO_SIZE[0] * PREVIEW_HEIGHT / ROCKET_VIDEO_SIZE[1]),
    PREVIEW_HEIGHT,
)
COMBINED_PREVIEW_SIZE = (
    PLOT_PREVIEW_SIZE[0] + ROCKET_PREVIEW_SIZE[0],
    PREVIEW_HEIGHT,
)
PREVIEW_FPS = 12


def _plot_figure(trajectory: Trajectory):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=tuple(dimension / 100 for dimension in PLOT_VIDEO_SIZE),
        dpi=100,
    )
    figure.suptitle("Dynamic Plot of Variables over Time", fontsize=14, y=0.98)
    angle_axis, rate_axis, force_axis, fuel_axis = axes.flat

    angle_lines = [
        angle_axis.plot([], [], linewidth=2, label=name)[0]
        for name in ("gamma", "psi", "phi")
    ]
    for axis, name in enumerate(("gamma_n", "psi_n", "phi_n")):
        angle_axis.plot(
            trajectory.time,
            trajectory.nominal_angles_deg[:, axis],
            "k--",
            linewidth=1.5,
            label=name,
        )
    angle_axis.axvline(130, color="k", linewidth=0.5)
    for value in (-3, 3, 117, 123):
        angle_axis.hlines(value, 129, 131, color="r", linewidth=1.5)
    angle_axis.set(
        title="Angles over Time",
        xlabel="Time Step (s)",
        ylabel="Angle (°)",
        xlim=(0, 131),
        ylim=(-10, 130),
    )
    angle_axis.legend(fontsize=8, loc="upper left")
    angle_axis.grid(True)

    rate_lines = [
        rate_axis.plot([], [], linewidth=2, label=name)[0]
        for name in ("gamma_dot", "psi_dot", "phi_dot")
    ]
    for time in (70, 130):
        rate_axis.axvline(time, color="k", linewidth=0.5)
        for value in (-1, -0.5, 0.5, 1):
            rate_axis.hlines(value, time - 1, time + 1, color="r", linewidth=1.5)
    rate_axis.set(
        title="Angle Velocities over Time",
        xlabel="Time Step (s)",
        ylabel="Angular Velocity (°/s)",
        xlim=(0, 131),
        ylim=(
            min(-1.2, float(trajectory.rates_deg_s.min()) - 0.1),
            max(1.2, float(trajectory.rates_deg_s.max()) + 0.1),
        ),
    )
    rate_axis.legend(fontsize=8, loc="upper left")
    rate_axis.grid(True)

    force_scatter = force_axis.scatter([], [], s=7, color="r")
    force_axis.set(
        title="Forces",
        xlabel="Time Step (s)",
        ylabel="Forces",
        xlim=(0, 131),
        ylim=(-0.5, 7.5),
    )
    force_axis.set_yticks(range(8), [f"F{i}" for i in range(1, 9)])
    force_axis.grid(True)

    fuel_line = fuel_axis.plot([], [], linewidth=2)[0]
    fuel_axis.set(
        title="Fuel",
        xlabel="Time Step (s)",
        ylabel="Fuel (N*s)",
        xlim=(0, 131),
        ylim=(0, max(1_000, float(trajectory.fuel_ns.max()) * 1.08)),
    )
    fuel_axis.grid(True)
    figure.tight_layout()
    figure.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
    return figure, angle_lines, rate_lines, force_scatter, fuel_line


def _plot_frame(artists, trajectory: Trajectory, end: int) -> np.ndarray:
    figure, angle_lines, rate_lines, force_scatter, fuel_line = artists
    for axis, line in enumerate(angle_lines):
        line.set_data(trajectory.time[:end], trajectory.angles_deg[:end, axis])
    for axis, line in enumerate(rate_lines):
        line.set_data(trajectory.time[:end], trajectory.rates_deg_s[:end, axis])
    active = trajectory.thrust_n[:end] != 0
    coordinates = np.column_stack(
        (
            np.repeat(trajectory.time[:end], 8)[active.ravel()],
            np.tile(np.arange(8), end)[active.ravel()],
        )
    )
    force_scatter.set_offsets(coordinates if len(coordinates) else np.empty((0, 2)))
    fuel_line.set_data(trajectory.time[:end], trajectory.fuel_ns[:end])
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def _state_at(trajectory: Trajectory, index: int) -> SimulationState:
    return SimulationState(
        time=float(trajectory.time[index]),
        angles=np.deg2rad(trajectory.angles_deg[index]),
        angular_rates=np.deg2rad(trajectory.rates_deg_s[index]),
        nominal_angles=np.deg2rad(trajectory.nominal_angles_deg[index]),
        nominal_rates=np.deg2rad(trajectory.nominal_rates_deg_s[index]),
        thrust=trajectory.thrust_n[index],
        moments=trajectory.moments_nm[index],
        fuel=float(trajectory.fuel_ns[index]),
        score=float(trajectory.rewards[: index + 1].sum()),
    )


def render_videos(
    trajectory: Trajectory,
    output_dir: str | Path,
    *,
    frames: int = 327,
    fps: int = 30,
    prefix: str = "a2c_2460000_seed0",
) -> tuple[Path, Path]:
    """Write synchronized plot and rocket-interface MP4 files."""
    if frames < 2 or fps <= 0:
        raise ValueError("frames must be at least 2 and fps must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plot_path = destination / f"{prefix}_dynamic_plot.mp4"
    rocket_path = destination / f"{prefix}_rocket_ui.mp4"
    indices = np.rint(np.linspace(0, len(trajectory.time) - 1, frames)).astype(int)
    artists = _plot_figure(trajectory)
    renderer = PygameRenderer(display=False)
    writer_options = {
        "fps": fps,
        "codec": "libx264",
        "pixelformat": "yuv420p",
        "macro_block_size": None,
        "ffmpeg_log_level": "warning",
    }
    with imageio.get_writer(plot_path, format="FFMPEG", **writer_options) as plot_writer, imageio.get_writer(
        rocket_path, format="FFMPEG", **writer_options
    ) as rocket_writer:
        for index in indices:
            plot_writer.append_data(_plot_frame(artists, trajectory, index + 1))
            rocket_writer.append_data(renderer.render(_state_at(trajectory, index)))
    renderer.close()
    plt.close(artists[0])
    return plot_path, rocket_path


def render_gif_preview(
    plot_video: str | Path,
    rocket_video: str | Path,
    *,
    fps: int = PREVIEW_FPS,
) -> Path:
    """Create one synchronized, side-by-side GIF for the repository README."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    plot_path = Path(plot_video)
    rocket_path = Path(rocket_video)
    prefix = plot_path.stem.removesuffix("_dynamic_plot")
    destination = plot_path.with_name(f"{prefix}_synchronized_preview.gif")
    plot_width, plot_height = PLOT_PREVIEW_SIZE
    rocket_width, rocket_height = ROCKET_PREVIEW_SIZE
    filters = (
        f"[0:v]fps={fps},scale={plot_width}:{plot_height}:flags=lanczos[plot];"
        f"[1:v]fps={fps},scale={rocket_width}:{rocket_height}:flags=lanczos[rocket];"
        "[plot][rocket]hstack=inputs=2:shortest=1,split[frames][palette_source];"
        "[palette_source]palettegen=max_colors=256:stats_mode=diff[palette];"
        "[frames][palette]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "warning",
            "-i",
            str(plot_path),
            "-i",
            str(rocket_path),
            "-filter_complex",
            filters,
            "-loop",
            "0",
            str(destination),
        ],
        check=True,
    )
    return destination
