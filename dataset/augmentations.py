from typing import Literal, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class TimeMask(nn.Module):
    """Zero a contiguous segment of the waveform (time masking, SpecAugment-style).

    Expects ``forward(waveform, sample_rate)`` and returns ``(waveform, sample_rate)``
    so it can be used as a ``SPEECHCOMMANDS_12C`` transform. When ``training`` is
    False, the input is returned unchanged.
    """

    def __init__(
        self,
        min_band_part: float = 0.01,
        max_band_part: float = 0.2,
        fade_duration: float = 0.005,
        mask_location: Literal["start", "end", "random"] = "random",
        p: float = 0.5,
    ) -> None:
        super().__init__()
        if not (0.0 <= min_band_part <= 1.0 and 0.0 <= max_band_part <= 1.0):
            raise ValueError("min_band_part and max_band_part must be in [0, 1]")
        if min_band_part > max_band_part:
            raise ValueError("min_band_part must not exceed max_band_part")
        if fade_duration < 0:
            raise ValueError("fade_duration must be non-negative")
        if mask_location not in {"start", "end", "random"}:
            raise ValueError('mask_location must be "start", "end", or "random"')

        self.min_band_part = min_band_part
        self.max_band_part = max_band_part
        self.fade_duration = fade_duration
        self.mask_location = mask_location
        self.p = p

    def forward(self, waveform: Tensor, sample_rate: int) -> Tuple[Tensor, int]:
        if not self.training or self.p <= 0 or torch.rand((), device=waveform.device) >= self.p:
            return waveform, sample_rate

        x = waveform
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        elif x.dim() != 2:
            raise ValueError("waveform must be (T,) or (C, T)")

        _, t_len = x.shape
        low = max(1, int(t_len * self.min_band_part))
        high = max(low, int(t_len * self.max_band_part))
        t = int(torch.randint(low, high + 1, (1,), device=x.device).item())

        loc = self.mask_location
        if loc == "start":
            t0 = 0
        elif loc == "end":
            t0 = t_len - t
        else:
            t0 = int(torch.randint(0, t_len - t + 1, (1,), device=x.device).item())

        out = x.clone()
        fade_len = 0
        if self.fade_duration > 0:
            fade_len = min(int(round(sample_rate * self.fade_duration)), t // 2)

        if fade_len >= 1:
            dtype, dev = out.dtype, out.device
            fade_in = torch.linspace(1, 0, fade_len, dtype=dtype, device=dev)
            fade_out = torch.linspace(0, 1, fade_len, dtype=dtype, device=dev)
            out[..., t0 : t0 + fade_len] *= fade_in
            mid_start, mid_end = t0 + fade_len, t0 + t - fade_len
            if mid_start < mid_end:
                out[..., mid_start:mid_end] = 0
            out[..., t0 + t - fade_len : t0 + t] *= fade_out
        else:
            out[..., t0 : t0 + t] = 0

        if squeeze:
            out = out.squeeze(0)
        return out, sample_rate
