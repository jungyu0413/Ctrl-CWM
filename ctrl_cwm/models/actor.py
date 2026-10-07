import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C
from .encoders import DSOC


def sample_local(local_map, pos):
    """Bilinear sample of local_map [A, c, G, G] at pos [(K,) A, 2] (grid px) -> [(K,) A, c]."""
    G = local_map.shape[-1]
    lead, A = pos.shape[:-2], pos.shape[-2]
    p = pos.reshape(-1, A, 2)
    grid = (p / (G - 1) * 2 - 1).permute(1, 0, 2)[:, :, None, :]
    local = F.grid_sample(local_map, grid, mode="bilinear", align_corners=True)
    return local.squeeze(-1).permute(2, 0, 1).reshape(*lead, A, -1)


class Actor(nn.Module):
    """pi_psi: [local scene feature, interaction embedding, goal direction, velocity / G] -> displacement."""

    def __init__(self, local_dim, dsoc=DSOC, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(local_dim + dsoc + 4, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, local, c, pos, goal, vel):
        d = goal - pos
        u = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
        return self.net(torch.cat([local, c, u, vel / C.GRID], -1))
