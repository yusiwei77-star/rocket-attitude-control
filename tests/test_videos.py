from pathlib import Path

import imageio.v2 as imageio
import pytest


VIDEOS = [
    Path("artifacts/demos/a2c_2460000_seed0_dynamic_plot.mp4"),
    Path("artifacts/demos/a2c_2460000_seed0_rocket_ui.mp4"),
]


@pytest.mark.parametrize("path", VIDEOS)
def test_published_video_metadata_and_frames(path: Path) -> None:
    reader = imageio.get_reader(path, format="FFMPEG")
    metadata = reader.get_meta_data()
    assert metadata["fps"] == pytest.approx(30.0)
    assert reader.count_frames() == 327
    for index in (0, 163, 326):
        frame = reader.get_data(index)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
    reader.close()
