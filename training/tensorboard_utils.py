from typing import Any, Dict

from logging import getLogger
import warnings
from matplotlib.figure import Figure
import torch
import matplotlib.pyplot as plt
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger


logger = getLogger(__name__)


def log_mel_spectrogram(
    writer: SummaryWriter,
    name: str,
    log_mel: torch.Tensor,
    label: str,
    step: int,
    predicted: str=None,
    x_axis: str = 'Time (s)',
    y_axis: str = 'Mels',
    cmap: str = 'viridis',
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    im = ax.imshow(log_mel.cpu().numpy(), aspect="auto", origin="lower", cmap=cmap)
    fig.colorbar(im, fraction=0.04, pad=0.01)
    title = f"Label: {label}"
    if predicted is not None:
        title += f" | Predicted: {predicted}"
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    image = render_figure(fig)
    writer.add_image(f"{name}", image, step)


def render_figure(fig: Figure) -> np.ndarray:
    with warnings.catch_warnings():
        # Squash UserWarning for missing from korean/Japanese/Chineses font.
        # ex. UserWarning: Glyph 4369 (\N{HANGUL CHOSEONG PHIEUPH}) missing from current font.
        warnings.filterwarnings("ignore", category=UserWarning)
        if not fig.get_constrained_layout():
            fig.tight_layout()
        try:
            fig.canvas.draw()
        except Exception as e:
            logger.exception(f"Could not draw figure {e}")
            del fig
            return np.zeros(())
        # tostring_rgb() was removed from some backends (e.g. macOS); buffer_rgba is the supported API.
        canvas = fig.canvas
        if hasattr(canvas, "buffer_rgba"):
            rgba = np.asarray(canvas.buffer_rgba())
            img = np.ascontiguousarray(rgba[..., :3])
        else:
            img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(
                canvas.get_width_height()[::-1] + (3,)
            )
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        plt.close(fig)
        del fig
    return img
