"""Check playback metadata in the exported artifact, not just file existence."""

import numpy as np
from PIL import Image

from aulecv import viz


def test_gif_preserves_frame_timing_colors_and_boomerang(tmp_path):
    colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    frames = [np.full((12, 16, 3), color, np.uint8) for color in colors_bgr]
    path = tmp_path / "playback.gif"
    viz.save_gif(str(path), frames, fps=4, boomerang=True)
    with Image.open(path) as gif:
        assert gif.n_frames == 4
        assert gif.info["loop"] == 0
        for index, rgb in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 0)]):
            gif.seek(index)
            assert gif.info["duration"] == 250
            assert gif.convert("RGB").getpixel((4, 4)) == rgb
