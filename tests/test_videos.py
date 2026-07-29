from pathlib import Path

import imageio.v2 as imageio
from PIL import Image
import pytest

from rocket_attitude_control.video import (
    COMBINED_PREVIEW_SIZE,
    PLOT_VIDEO_SIZE,
    PREVIEW_FPS,
    ROCKET_VIDEO_SIZE,
)

VIDEOS = [
    (Path("artifacts/demos/a2c_2460000_seed0_dynamic_plot.mp4"), PLOT_VIDEO_SIZE),
    (Path("artifacts/demos/a2c_2460000_seed0_rocket_ui.mp4"), ROCKET_VIDEO_SIZE),
]

PREVIEW = Path("artifacts/demos/a2c_2460000_seed0_synchronized_preview.gif")


@pytest.mark.parametrize(("path", "expected_size"), VIDEOS)
def test_published_video_metadata_and_frames(
    path: Path,
    expected_size: tuple[int, int],
) -> None:
    reader = imageio.get_reader(path, format="FFMPEG")
    metadata = reader.get_meta_data()
    assert metadata["size"] == expected_size
    assert metadata["fps"] == pytest.approx(30.0)
    assert reader.count_frames() == 327
    for index in (0, 163, 326):
        frame = reader.get_data(index)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.shape[:2] == expected_size[::-1]
    reader.close()


def test_readme_preview_metadata() -> None:
    with Image.open(PREVIEW) as preview:
        assert preview.size == COMBINED_PREVIEW_SIZE
        frame_count = preview.n_frames
        assert frame_count == 131
        duration_seconds = sum(
            preview.seek(index) or preview.info["duration"]
            for index in range(frame_count)
        ) / 1000
    assert duration_seconds == pytest.approx(327 / 30, abs=0.05)
    assert frame_count / duration_seconds == pytest.approx(PREVIEW_FPS, abs=0.05)
