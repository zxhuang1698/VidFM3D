import torch
import torch.nn as nn


class Modulation(nn.Module):
    """
    FiLM‑style conditioning:  x <- x * (1 + scale) + shift,
    where [scale, shift] is a linear projection of an external vector `cond`.
    """

    def __init__(
        self,
        embed_dim: int,  # same as the LayerNorm dim
        cond_dim: int,  # size of conditioning vector
        zero_init: bool = True,
        single_layer: bool = False,
    ):
        super().__init__()
        self.act = nn.SiLU()
        self.fc1 = nn.Identity() if single_layer else nn.Linear(cond_dim, cond_dim)
        self.fc2 = nn.Linear(cond_dim, 2 * embed_dim)
        if zero_init:  # start as a no‑op
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.fc2(self.act(self.fc1(cond))).chunk(2, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
