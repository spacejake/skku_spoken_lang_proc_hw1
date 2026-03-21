import torch
from torch import autocast
from torch import nn
import einops
import logging


logger = logging.getLogger(__name__)

class FixedParameterizationLayer:
    """Mixin class that prevents weight normalization and other reparameterizations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exclude_reparameterization = True


def apply_weight_norm(model: nn.Module):
    """Apply weight normalization module from all of the layers."""

    def _apply_weight_norm(m):
        # exclude some modules from weight norm
        if isinstance(m, FixedParameterizationLayer):
            logger.debug(f"Weight norm is not applied to {m}.")
            return
        
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            nn.utils.parametrizations.weight_norm(m)
            logger.debug(f"Weight norm is applied to {m}.")

    model.apply(_apply_weight_norm)


def remove_weight_norm(model: nn.Module):
    """Remove weight normalization module from all of the layers."""
    def _remove_weight_norm(m):
        if isinstance(m, FixedParameterizationLayer):
            return
        
        try:
            logger.debug(f"Weight norm is removed from {m}.")
            nn.utils.parameterize.remove_parametrization(m, "weight")
        except ValueError:  # this module didn't have weight norm
            return
    
    model.apply(_remove_weight_norm)



def mask_sequence_tensor(tensor: torch.Tensor, lengths: torch.Tensor):
    """
    For tensors containing sequences, zero out out-of-bound elements given lengths of every element in the batch.

    tensor: tensor of shape (B, L), (B, D, L) or (B, D1, D2, L),
    lengths: LongTensor of shape (B,)
    """
    batch_size, *_, max_lengths = tensor.shape

    if len(tensor.shape) == 2:
        mask = torch.ones(batch_size, max_lengths, dtype=lengths.dtype, device=lengths.device).cumsum(dim=-1)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1')
    elif len(tensor.shape) == 3:
        mask = torch.ones(batch_size, 1, max_lengths, dtype=lengths.dtype, device=lengths.device).cumsum(dim=-1)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1 1')
    elif len(tensor.shape) == 4:
        mask = torch.ones(batch_size, 1, 1, max_lengths, dtype=lengths.dtype, device=lengths.device).cumsum(dim=-1)
        mask = mask <= einops.rearrange(lengths, 'B -> B 1 1 1')
    else:
        raise ValueError('Can only mask tensors of shape B x L, B x D x L and B x D1 x D2 x L')

    return tensor * mask