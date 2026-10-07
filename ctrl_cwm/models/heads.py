import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import cbr


def _up(x, ref):
    return F.interpolate(x, size=ref.shape[-1], mode="nearest")


class GoalDecoder(nn.Module):
    """Modulated bottleneck + scene skips -> goal heatmap M_goal (endpoint of the L-step window)."""

    def __init__(self, w):
        super().__init__()
        self.d4 = cbr(w(256) + w(256), w(128)); self.d3 = cbr(w(128) + w(128), w(64))
        self.d2 = cbr(w(64) + w(64), w(32)); self.d1 = cbr(w(32) + w(32), w(32))
        self.head = nn.Conv2d(w(32), 1, 1)

    def forward(self, b, skips):
        s1, s2, s3, s4 = skips
        x = self.d4(torch.cat([_up(b, s4), s4], 1))
        x = self.d3(torch.cat([_up(x, s3), s3], 1))
        x = self.d2(torch.cat([_up(x, s2), s2], 1))
        x = self.d1(torch.cat([_up(x, s1), s1], 1))
        return self.head(x)[:, 0]


class WaypointDecoder(nn.Module):
    """Goal-conditioned trunk shared by the waypoint head (L heatmaps) and the auxiliary dynamics head."""

    def __init__(self, w, horizon):
        super().__init__()
        self.d4 = cbr(w(256) + w(256) + w(128), w(128)); self.d3 = cbr(w(128) + w(128) + w(64), w(64))
        self.d2 = cbr(w(64) + w(64) + w(32), w(32)); self.d1 = cbr(w(32) + w(32) + w(16), w(32))
        self.waypoint = nn.Conv2d(w(32), horizon, 1)
        self.dynamics = nn.Conv2d(w(32), 1, 1)

    def forward(self, b, skips, gskips):
        s1, s2, s3, s4 = skips
        g1, g2, g3, g4 = gskips
        x = self.d4(torch.cat([_up(b, s4), s4, g4], 1))
        x = self.d3(torch.cat([_up(x, s3), s3, g3], 1))
        x = self.d2(torch.cat([_up(x, s2), s2, g2], 1))
        x = self.d1(torch.cat([_up(x, s1), s1, g1], 1))
        return self.waypoint(x), self.dynamics(x)[:, 0]
