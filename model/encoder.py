from typing import NamedTuple, Tuple
import math
import logging
import time
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
import torchaudio


try:
    import model
except ImportError:
    import sys
    import git
    git_repo_root = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
    if git_repo_root not in sys.path:
        sys.path.append(git_repo_root)

from model.activations import Snake
from model.modules import ConvNeXtBlock, TransposedLayerNorm
from model.utils import apply_weight_norm


logger = logging.getLogger(__name__)


def get_down_sample_padding(kernel_size: int, stride: int) -> int:
    return (kernel_size - stride + 1) // 2


class BottleneckBiGRU(nn.Module):
    def __init__(self, channels: int, hidden: int = 256, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.in_ln  = nn.LayerNorm(channels)
        self.gru    = nn.GRU(
            input_size=channels, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0.0
        )
        self.proj   = nn.Linear(2 * hidden, channels)   # back to original C
        # light residual gating
        self.gamma  = nn.Parameter(torch.ones(1) * 1e-3)

    def forward(self, x):             # x: [B, C, T]
        x0 = x
        x = x.transpose(1, 2)         # [B, T, C]
        x = self.in_ln(x)
        y, _ = self.gru(x)            # [B, T, 2H]
        y = self.proj(y)              # [B, T, C]
        y = y.transpose(1, 2)         # [B, C, T]
        return x0 + self.gamma * y    # residual, safe init


class ConvNeXtStage(nn.Module):
    def __init__(
        self,
        hdim: int = 512,
        intermediate_dim: int = 1024,
        kernel_sizes: list[int] = [3, 5, 7],
        dilations: list[int] = [1, 3, 5],
        causal: bool = False,
        use_grn: bool = True,
        act: Literal["gelu", "silu", "snake"] = "gelu",
    ):
        super().__init__()

        ds_layer_scale = 1 / len(kernel_sizes)

        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=hdim,
                    intermediate_dim=intermediate_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    causal=causal,
                    layer_scale_init_value=ds_layer_scale,
                    use_grn=use_grn,
                    act=act,
                ) for kernel_size, dilation in zip(kernel_sizes, dilations)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_block in self.convnext:
            x = conv_block(x)
        return x

class Encoder(nn.Module):
    """Expects raw waveform; the first learned stage sees a log-mel spectrogram (dB)."""

    def __init__(
        self,
        # Defaults sized for <~2.5M trainable params (Speech Commands–scale).
        hdim: int = 128,
        odim: int = 128,
        kernel_size_feat: int = 7,
        encoder_kernel_sizes: list[int] = [7, 7, 7],
        encoder_dilations: list[int] = [1, 3, 5],
        kernel_sizes: list[int] = [7, 7, 7],
        # Milder than [1,3,5] after 4× time stride so 1s clips keep enough frames for reflect padding.
        dilation_list: list[int] = [1, 2, 3],
        intermediate_dim: int = 384,
        n_fft: int = 1024,
        # Spectrogram at 50Hz frame rate
        win_length: int = 640,  # 40ms = 16000 × 0.04
        hop_length: int = 320,  # 20ms = 16000 × 0.02
        n_mels: int = 80,
        sample_rate: int = 16000,
        dropout: float = 0.0,
        act: Literal["gelu", "lrelu", "silu", "snake"] = "snake",
        # TO get to 12.5kHz, we need to downsample by 2x twice at 50Hz frame rate
        downsample_kernel_sizes: list[int] = [7, 5],
        downsample_factors: list[int] = [2, 2],
        downsample_dims: list[int] = [128, 128],
        use_weight_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert len(downsample_factors) == len(downsample_kernel_sizes)
        assert len(kernel_sizes) == len(dilation_list)
        assert kernel_size_feat % 2 == 1
        assert len(encoder_kernel_sizes) == len(encoder_dilations)
        self.odim = odim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.downsample_factors = downsample_factors
        self.downsample_kernel_sizes = downsample_kernel_sizes
        self.downsample_dims = downsample_dims
        self.kernel_sizes = kernel_sizes
        self.dilation_list = dilation_list
        self.encoder_kernel_sizes = encoder_kernel_sizes
        self.encoder_dilations = encoder_dilations

        f_max = sample_rate / 2.0  # nyquist frequency
        self.log_mel = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                n_mels=n_mels,
                f_min=0.0,
                f_max=f_max,
            ),
            torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0),
        )

        # Compute frame rate correctly
        self.frame_rate = sample_rate / hop_length  # frames per second

        self.embed = nn.Sequential(
            nn.Conv1d(
                n_mels,
                hdim,
                kernel_size_feat,
                padding=int(kernel_size_feat / 2),
                padding_mode="reflect",
            ),
            TransposedLayerNorm(hdim),
            Snake(hdim),
        )

        self.embed_convnext = ConvNeXtStage(
            hdim=hdim,
            intermediate_dim=intermediate_dim,
            kernel_sizes=encoder_kernel_sizes,
            dilations=encoder_dilations,
            causal=False,
            use_grn=True,
            act=act,
        )

        frame_rate_at_stage = self.frame_rate
        prev_frame_rate = frame_rate_at_stage
        self.post_downsample_conv_blocks = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()
        pre_downsample_dim = hdim
        for downsample_kernel_size, downsample_factor, downsample_dim in zip(downsample_kernel_sizes, downsample_factors, downsample_dims):
            # self.activations.append(get_activation(act, channels=hdim))
            # downsample conv
            padding = get_down_sample_padding(kernel_size=downsample_kernel_size, stride=downsample_factor)
            self.downsample_convs.append(nn.Conv1d(
                in_channels=pre_downsample_dim,
                out_channels=downsample_dim,
                kernel_size=downsample_kernel_size,
                stride=downsample_factor,
                padding=padding, padding_mode="reflect",
            ))

            # Lightweight learned block at cheaper frame rate after decimation
            self.post_downsample_conv_blocks.append(
                ConvNeXtStage(
                    hdim=downsample_dim,
                    intermediate_dim=intermediate_dim,
                    kernel_sizes=kernel_sizes,
                    dilations=dilation_list,
                    causal=False,
                    use_grn=True,
                    act=act,
                )
            )
            frame_rate_at_stage = frame_rate_at_stage / downsample_factor
            logger.debug(
                f"""downsample factor: {downsample_factor}, """
                f"""frame rate at stage: {prev_frame_rate} -> {frame_rate_at_stage}, """
                f"""pre_proj_dim: {pre_downsample_dim} -> {downsample_dim}"""
            )
            prev_frame_rate = frame_rate_at_stage
            pre_downsample_dim = downsample_dim

        self.frame_rate = frame_rate_at_stage  # final frame rate
        downsample_hdim = pre_downsample_dim
        self.recurrent_block = BottleneckBiGRU(downsample_hdim, hidden=64, layers=1, dropout=dropout)
        self.proj = nn.Conv1d(downsample_hdim, odim, 1, bias=False)
        self.final_norm = TransposedLayerNorm(odim)

        if use_weight_norm:
            self.apply_weight_norm()
        self.reset_parameters()

    def apply_weight_norm(self):
        apply_weight_norm(self)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """
        def _reset_parameters(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logger.debug(f"Reset parameters in {m}.")
        self.apply(_reset_parameters)

    @property
    def downsample_factor(self) -> int:
        return self.hop_length * math.prod(self.downsample_factors)

    @property
    def output_dim(self) -> int:
        return self.odim

    @torch.no_grad()
    def to_spec(self, wav: torch.Tensor) -> torch.Tensor:
        return self.log_mel(wav)

    def pad_to_frame_length(self, wav: torch.Tensor, mode: str = "reflect", **kwargs) -> torch.Tensor:
        # Pad input to ensure STFT produces frames that are divisible by downsample factor
        # wav: [B, T]
        N = wav.size(-1)
        T = 1 + (N // self.hop_length)  # with center=True
        need_frames = (-T) % math.prod(self.downsample_factors)  # how many frames to add
        pad_samples = need_frames * self.hop_length
        if pad_samples:
            wav = torch.nn.functional.pad(wav, (0, pad_samples), mode=mode)
        return wav

    def forward(self, wav: torch.Tensor, return_intermediates: bool = False,) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        log_mel_spec = self.log_mel(wav)
        x = self.embed(log_mel_spec)
        x = self.embed_convnext(x)

        for downsample, post_blk in zip(
            self.downsample_convs,
            self.post_downsample_conv_blocks,
        ):
            x = downsample(x)
            x = post_blk(x)

        x = self.recurrent_block(x)
        x = self.proj(x)
        x = self.final_norm(x)

        if return_intermediates:
            return x, log_mel_spec

        return x


if __name__ == "__main__":
    # Get available devices

    if torch.cuda.is_available():
        device = 'cuda:0'
        print("CUDA device found. Using GPU.")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("MPS device found. Using GPU.")
    else:
        device = torch.device("cpu")
        print("MPS device not found. Using CPU.")

    model = Encoder().to(device)

    print(model)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Test the forward function
    wav = torch.randn(1, 16000).to(device)
    x = model(wav)
    print(x.shape)

    # Test the to_spec function
    spec = model.to_spec(wav)
    print(spec.shape)

    # Test the pad_to_frame_length function
    wav = torch.randn(1, 16000).to(device)
    wav = model.pad_to_frame_length(wav)
    print(wav.shape)