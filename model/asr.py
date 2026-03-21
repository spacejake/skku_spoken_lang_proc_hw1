from __future__ import annotations

from typing import Literal

import torch
from torch import nn

try:
    import model
except ImportError:
    import sys
    import git

    git_repo_root = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
    if git_repo_root not in sys.path:
        sys.path.append(git_repo_root)

from model.encoder import Encoder


class SpeechCommandClassifier(nn.Module):
    """Encoder over raw waveform, time pooling, and MLP head for keyword classes."""

    def __init__(
        self,
        num_classes: int = 12,
        encoder: Encoder | None = None,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.2,
        pooling: Literal["mean", "max"] = "mean",
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else Encoder(**encoder_kwargs)
        embed_dim = self.encoder.output_dim
        if pooling == "mean":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"pooling must be 'mean' or 'max', got {pooling!r}")
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, wav: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # wav: [B, num_samples]
        h = self.encoder(wav, return_intermediates=return_intermediates)  # [B, embed_dim, num_frames]
        if return_intermediates:
            h, log_mel_spec = h
        h = self.pool(h).squeeze(-1)  # [B, embed_dim]
        logits = self.head(h)  # [B, num_classes]
        if return_intermediates:
            return logits, log_mel_spec
        return logits


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

    model = SpeechCommandClassifier().to(device)
    print(model)
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Test the forward function
    wav = torch.randn(1, 16000).to(device)
    x = model(wav)
    print(x.shape)
