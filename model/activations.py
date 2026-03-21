import torch
from torch import nn
import torch.nn.functional as F

try:
    import model
except ImportError:
    import sys
    import git
    git_repo_root = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
    if git_repo_root not in sys.path:
        sys.path.append(git_repo_root)

from model.globals import EPS, HP_EPS


@torch.jit.script
def snake_cos(x: torch.Tensor, alpha: torch.Tensor, eps: float = EPS) -> torch.Tensor:
   """
   Snake activation using cosine formulation: x + (1 - cos(2αx)) / (2α + ε)
   """
   return x + (1 - torch.cos(2 * alpha * x)) / (2 * alpha + eps)


class Snake(nn.Module):
    """
    Snake activation function introduced in 'https://arxiv.org/abs/2006.08195'
    """

    def __init__(self, channels: int, eps: float = EPS, channel_last: bool = False, softplus: bool = True):
        super().__init__()
        if channel_last:
            self.alpha = nn.Parameter(torch.ones(1, 1, channels))
        else:
            self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.eps = eps
        self.softplus = softplus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # enforce positive alpha
        if self.softplus:
            alpha = F.softplus(self.alpha) + self.eps
        return snake_cos(x, self.alpha, self.eps)


def get_activation(activation: str = "lrelu", channels: int = 1) -> nn.Module:
    """
    Choose between activation based on the input parameter.

    Args:
        activation: Name of activation to use. Valid options are "elu" (default), "lrelu", and "snake".
        channels: Input dimension.
    """

    activation = activation.lower()
    if activation == "elu":
        return nn.ELU()
    elif activation == "lrelu":
        return torch.nn.LeakyReLU()
    elif activation == "snake":
        return Snake(channels)
    else:
        raise ValueError(f"Unknown activation {activation}")
