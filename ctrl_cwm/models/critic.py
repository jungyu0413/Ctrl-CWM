import torch
import torch.nn as nn

from .. import config as C

CRITIC_FEATS = 6


class Critic(nn.Module):
    """V_phi: score of one agent's motion from its geometric features over H imagined steps."""

    def __init__(self, cin=CRITIC_FEATS, hid=128):
        super().__init__()
        self.gru = nn.GRU(cin, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.LeakyReLU(0.2), nn.Linear(hid, 1))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class Discriminator(nn.Module):
    """LSGAN discriminator over a normalized displacement sequence [A, L, 2]."""

    def __init__(self, hid=128):
        super().__init__()
        self.gru = nn.GRU(2, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.LeakyReLU(0.2), nn.Linear(hid, 1))

    def forward(self, vel):
        _, h = self.gru(vel)
        return self.head(h[-1]).squeeze(-1)


def critic_features(pos, vel, goal):
    """pos, vel [W, A, H, 2] of W imagined worlds, goal [(W,) A, 2] ->
    [vx, vy, d_nbr, d_goal, cos(v, g - p), |v|] / G per agent and step, [W, A, H, 6]."""
    W, A, H, _ = pos.shape
    eye = torch.eye(A, device=pos.device) * 1e6
    d_nbr = torch.stack([(torch.cdist(pos[:, :, t], pos[:, :, t]) + eye).min(-1).values for t in range(H)], -1)
    gvec = goal[..., None, :] - pos
    d_goal = gvec.norm(dim=-1)
    speed = vel.norm(dim=-1)
    cos = (vel * gvec).sum(-1) / ((d_goal + 1e-6) * (speed + 1e-6))
    G = C.GRID
    return torch.stack([vel[..., 0] / G, vel[..., 1] / G, d_nbr / G, d_goal / G, cos, speed / G], -1)
