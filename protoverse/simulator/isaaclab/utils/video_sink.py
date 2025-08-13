import imageio
import numpy as np
import torch


def to_uint8_rgb(x) -> np.ndarray:
    """Accept torch/np, shape (...,H,W,3) or (H,W,3); return np.uint8 (H,W,3)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 4:
            x = x[0]
        arr = x.numpy()
    else:
        arr = np.asarray(x)
        if arr.ndim == 4:
            arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class VideoSink:
    """Simple streaming MP4 writer (libx264, yuv420p)."""

    def __init__(self, mp4_path: str, fps: int = 30):
        self._mp4_path = mp4_path
        self._fps = fps
        self._writer = None  # lazy-open on first frame

    def _open(self):
        if self._writer is None:
            # Open with ffmpeg; tune similarly to your old settings
            self._writer = imageio.get_writer(
                self._mp4_path,
                format="FFMPEG",
                fps=self._fps,
                codec="libx264",
                pixelformat="yuv420p",
                ffmpeg_params=[
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-movflags",
                    "+faststart",
                    "-profile:v",
                    "main",
                    "-level",
                    "4.0",
                ],
            )

    def append(self, rgb_uint8: np.ndarray):
        """rgb_uint8: HxWx3, np.uint8"""
        if self._writer is None:
            self._open()
        self._writer.append_data(rgb_uint8)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
