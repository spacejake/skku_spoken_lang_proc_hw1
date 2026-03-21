from typing import Literal, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from model.globals import EPS, HP_EPS
from model.activations import Snake


class TransposedLayerNorm(nn.Module):
    def __init__(self, idim: int, eps: float = HP_EPS):
        super().__init__()
        self.norm = nn.LayerNorm(idim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [b, c, t]
        return self.norm(x.transpose(1, 2)).transpose(1, 2).contiguous()

class CausalConv1d(nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        ksz: int,
        strd: int = 1,
        dilation: int = 1,
        padding_mode: str = "reflect",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.net = nn.Conv1d(idim, odim, ksz, strd, dilation=dilation, *args, **kwargs)
        self.padding_mode = padding_mode
        self.n_pad = n_pad = dilation * (ksz - 1)
        self.register_buffer("stream_buffer", torch.zeros(1, idim, n_pad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode == "replicate":
            x_pad = F.pad(x, (self.n_pad, 0), mode="replicate")
        elif self.padding_mode == "zeros":
            x_pad = F.pad(x, (self.n_pad, 0), value=0)
        elif self.padding_mode == "reflect":
            x_pad = F.pad(x, (self.n_pad, 0), mode="reflect")
        else:
            raise ValueError(f"Invalid padding mode: {self.padding_mode}")
        out = self.net(x_pad)
        return out


def get_activation(
    act: Literal["gelu", "lrelu", "silu", "snake"] = "gelu",
    channels: int | None = None,
    slope: float = 0.1,
) -> nn.Module:
    """Get activation function based on the input parameter."""
    if act == "gelu":
        return nn.GELU()
    elif act == "lrelu":
        return nn.LeakyReLU(slope)
    elif act == "silu":
        return nn.SiLU()
    elif act == "snake":
        return Snake(channels)
    else:
        raise ValueError(f"Invalid activation function: {act}")


class LayerScaledMLP(nn.Module):
    """Layer scaled pointwise convolution. Used for stacking ConvNeXtV1 blocks."""
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        act: Literal["gelu", "lrelu", "silu", "snake"] = "gelu",
    ):
        super().__init__()
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, 1)  # pointwise conv
        self.act = get_activation(act, channels=intermediate_dim)
        self.pwconv2 = nn.Conv1d(intermediate_dim, dim, 1)  # pointwise conv
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(1, dim, 1), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer for 1D convolutions."""

    def __init__(self, dim):
        super().__init__()
        # Learnable scaling and shifting parameters
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))  # (1, C, 1) for 1D convolutions
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))  # (1, C, 1) for 1D convolutions

    def forward(self, x):
        # Gx is the global L2 norm across the temporal dimension (dim=2) for each channel
        Gx = torch.norm(x, p=2, dim=2, keepdim=True)  # Norm over time axis (dim=2)

        # Normalize Gx over the mean in the channel axis (dim=1)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + EPS)  # Normalize within channels

        # Apply learned scale (gamma) and shift (beta), and add skip connection
        return self.gamma * (x * Nx) + self.beta + x


class GRNMLP(nn.Module):
    """GRN (Global Response Normalization) layer for 1D convolutions. Used in ConvNeXtV2."""
    def __init__(self, dim: int, intermediate_dim: int, act: Literal["gelu", "lrelu", "silu", "snake"] = "gelu"):
        super().__init__()
        
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, 1)  # pointwise conv
        self.act = get_activation(act, channels=intermediate_dim)
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Conv1d(intermediate_dim, dim, 1)  # pointwise conv
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return x
    

class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        dilation: int = 1,
        causal: bool = False,
        use_grn: bool = False,
        act: Literal["gelu", "lrelu", "silu", "snake"] = "gelu",
    ):
        super().__init__()
        if causal:
            self.dwconv = CausalConv1d(dim, dim, kernel_size, dilation=dilation, groups=dim, padding_mode="reflect")
        else:
            padding = int((dilation * (kernel_size - 1)) / 2)
            self.dwconv = nn.Conv1d(
                dim, dim, kernel_size, padding=padding, dilation=dilation, groups=dim, padding_mode="reflect"
            )
        self.norm = TransposedLayerNorm(dim)
        if use_grn:
            self.mlp = GRNMLP(dim, intermediate_dim, act)
        else:
            self.mlp = LayerScaledMLP(dim, intermediate_dim, layer_scale_init_value, act)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask
        residual = x
        x = self.dwconv(x)
        if mask is not None:
            x = x * mask
        x = self.norm(x)
        x = self.mlp(x)
        
        x = residual + x
        if mask is not None:
            x = x * mask
        return x

# --- TinyVocos Lightweight Attention ----------------------

# ---------- tiny norms ----------
class RMSNorm1d(nn.Module):
    """RMSNorm over channel dim for (B, C, T)."""
    def __init__(self, C: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, C, 1))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_inv = (x.pow(2).mean(dim=1, keepdim=True) + self.eps).rsqrt()
        return self.weight * x * rms_inv


# ---------- Snake-XiConv (1D) core ----------


class SnakeXiConv1d(nn.Module):
    """
    1D TinyVocos-style XiConv with Snake activation and lite attention gate.
    All pointwise 'linears' are Conv1d(1x1). No transposes.
    """
    def __init__(
        self,
        dim: int,
        odim: int,
        kernel_size: int,
        dilation: int = 1,
        compression: int = 4,
        attention_k: int = 9,
        causal: bool = False,
        padding_mode: str = "reflect",
        dropout: float = 0.0,
        act: Literal["snake", "silu"] = "snake",
    ):
        super().__init__()
        assert compression >= 1
        hidden = max(1, dim // compression)
        self.compression = compression

        # bottleneck in/out (1x1 convs)
        if compression > 1:
            self.pw_in  = nn.Conv1d(dim, hidden, 1, bias=False)
        else:
            hidden = dim

        if causal:
            self.dwconv = CausalConv1d(hidden, dim, kernel_size, dilation=dilation, groups=dim, padding_mode=padding_mode)
        else:
            padding = int((dilation * (kernel_size - 1)) / 2)
            self.dwconv = nn.Conv1d(
                hidden, dim, kernel_size, padding=padding, dilation=dilation, groups=hidden, padding_mode=padding_mode
            )

        # norm + Snake
        self.norm = TransposedLayerNorm(dim)
        if act == "snake":
            self.act = Snake(dim)
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Invalid activation function: {act}")

        self.att_in  = nn.Conv1d(dim, hidden, 1, bias=False)
        att_padding = int((attention_k - 1) / 2)
        self.att_conv= nn.Conv1d(hidden, dim, attention_k, padding=att_padding, bias=False, padding_mode="reflect")
        self.att_act = nn.Sigmoid()

        self.dropout = dropout
        if self.dropout > 0:
            self.do = nn.Dropout(dropout)
        
        if odim != dim:
            self.proj = nn.Conv1d(dim, odim, 1, bias=False)
        else:
            self.proj = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        # bottleneck -> temporal (depthwise) -> expand
        if self.compression > 1:
            h = self.pw_in(x)          # (B,H,T)
        else:
            h = x
        h = self.dwconv(h)       # (B,C,T)

        # norm + snake on the mixed features
        h = self.act(self.norm(h))

        # attention-like gate
        g = self.att_act(self.att_conv(F.silu(self.att_in(x))))  # (B,C,T)
        h = h * g

        if self.dropout > 0:
            h = self.do(h)
        
        x = x + h
        
        if self.proj is not None:
            x = self.proj(x)
        return x


# ---------- DROP-IN: ConvNeXtBlock replacement ----------
class SnakeXiConvNeXtBlock(nn.Module):
    """
    Drop-in replacement for your ConvNeXtBlock (1D, (B,C,T) tensors).
    Inside uses SnakeXiConv1d + 1x1 Conv MLP (LayerScaled or GRN).
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        dilation: int = 1,
        causal: bool = False,
        use_grn: bool = False,
        # --- extras to control XiConv behavior (kept defaulted for "drop-in") ---
        compression: int = 4,
        attention_k: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temporal = SnakeXiConv1d(
            dim=dim,
            odim=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            compression=compression,
            attention_k=attention_k,
            causal=causal,
            dropout=dropout,
            act="silu",
        )

        # MLP head (1x1 convs only)
        if use_grn:
            self.mlp = GRNMLP(dim, intermediate_dim)
        else:
            self.mlp = LayerScaledMLP(dim, intermediate_dim, layer_scale_init_value, act="silu")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:    (B, C, T)
        mask: (B, 1, T) or (B, C, T) with {0,1}, optional
        """
        if mask is not None:
            x = x * mask

        # Snake-Xi temporal block (includes residual)
        x = self.temporal(x)

        if mask is not None:
            x = x * mask

        # 1x1 MLP; residual like original ConvNeXtBlock
        residual = x
        x = self.mlp(x)
        x = residual + x

        if mask is not None:
            x = x * mask

        return x



class ResBlock1(nn.Module):
    """
    ResBlock adapted from HiFi-GAN V1 (https://github.com/jik876/hifi-gan) with dilated 1D convolutions,
    but without upsampling layers.

    Args:
        dim (int): Number of input channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        dilation (tuple[int], optional): Dilation factors for the dilated convolutions.
            Defaults to (1, 3, 5).
        lrelu_slope (float, optional): Negative slope of the LeakyReLU activation function.
            Defaults to 0.1.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
        lrelu_slope: float = 0.1,
        layer_scale_init_value: float | None = None,
    ):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=self.get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=self.get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=self.get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
                weight_norm(nn.Conv1d(dim, dim, kernel_size, 1, dilation=1, padding=self.get_padding(kernel_size, 1))),
            ]
        )

        self.gamma = nn.ParameterList(
            [
                (
                    nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                    if layer_scale_init_value is not None
                    else None
                ),
                (
                    nn.Parameter(layer_scale_init_value * torch.ones(dim, 1), requires_grad=True)
                    if layer_scale_init_value is not None
                    else None
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, gamma in zip(self.convs1, self.convs2, self.gamma):
            xt = torch.nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = c1(xt)
            xt = torch.nn.functional.leaky_relu(xt, negative_slope=self.lrelu_slope)
            xt = c2(xt)
            if gamma is not None:
                xt = gamma * xt
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

    @staticmethod
    def get_padding(kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)