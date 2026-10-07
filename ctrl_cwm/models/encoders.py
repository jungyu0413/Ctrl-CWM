from dataclasses import dataclass

import torch
import torch.nn as nn

SEG_C = 9
DSOC = 96


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())


class InteractionEncoder(nn.Module):
    """Attention over co-present agents with pairwise-displacement biases, and a GRU over velocity history."""

    def __init__(self, ad=64, dsoc=DSOC):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(2, ad), nn.ReLU())
        self.relbias = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.attn = nn.MultiheadAttention(ad, 4, batch_first=True)
        self.velgru = nn.GRU(2, ad, batch_first=True)
        self.out = nn.Sequential(nn.Linear(ad + ad, dsoc), nn.ReLU(), nn.Linear(dsoc, dsoc))

    def forward(self, pos, vel_hist, allow):
        """pos [(W,) A, 2], vel_hist [(W,) A, T, 2] (grid / GRID), allow [A, A] additive mask -> [(W,) A, DSOC]."""
        batched = pos.dim() == 3
        p = pos if batched else pos[None]
        W, A, _ = p.shape
        rp = p[:, None, :, :] - p[:, :, None, :]
        mask = (self.relbias(rp).squeeze(-1) + allow).repeat_interleave(self.attn.num_heads, 0)
        t = self.tok(p)
        soc, _ = self.attn(t, t, t, attn_mask=mask)
        _, hv = self.velgru(vel_hist.reshape(-1, *vel_hist.shape[-2:]))
        hv = hv[0].reshape(*vel_hist.shape[:-2], -1)
        hv = hv.expand(W, -1, -1) if hv.dim() == 2 else hv
        out = self.out(torch.cat([soc, hv], -1))
        return out if batched else out[0]


class SceneEncoder(nn.Module):
    """U-Net encoder over [scene, focal heatmap, crowd heatmap]; the bottleneck is modulated by FiLM."""

    def __init__(self, w, cin=SEG_C + 2, dsoc=DSOC):
        super().__init__()
        self.e1 = cbr(cin, w(32)); self.e2 = cbr(w(32), w(64)); self.e3 = cbr(w(64), w(128)); self.e4 = cbr(w(128), w(256))
        self.pool = nn.MaxPool2d(2)
        self.bott = cbr(w(256), w(256))
        self.film = nn.Linear(dsoc, w(256) * 2)

    def forward(self, x, c):
        s1 = self.e1(x); s2 = self.e2(self.pool(s1)); s3 = self.e3(self.pool(s2)); s4 = self.e4(self.pool(s3))
        b = self.bott(self.pool(s4))
        g, be = self.film(c).chunk(2, -1)
        return b * (1 + g[:, :, None, None]) + be[:, :, None, None], (s1, s2, s3, s4)


class GoalEncoder(nn.Module):
    def __init__(self, w, cin=1):
        super().__init__()
        self.g1 = cbr(cin, w(16)); self.g2 = cbr(w(16), w(32)); self.g3 = cbr(w(32), w(64)); self.g4 = cbr(w(64), w(128))
        self.pool = nn.MaxPool2d(2)

    def forward(self, goal_hm):
        x = goal_hm[:, None] if goal_hm.dim() == 3 else goal_hm
        g1 = self.g1(x); g2 = self.g2(self.pool(g1)); g3 = self.g3(self.pool(g2)); g4 = self.g4(self.pool(g3))
        return g1, g2, g3, g4


@dataclass
class Features:
    """Agent-centric representation z: modulated bottleneck, scene skips, interaction embedding."""
    bottleneck: torch.Tensor
    skips: tuple
    interaction: torch.Tensor

    @property
    def local_map(self):
        return self.skips[0]


class FeatureExtractor(nn.Module):
    """h_theta."""

    def __init__(self, w, use_interaction=True):
        super().__init__()
        self.use_interaction = use_interaction
        self.interaction = InteractionEncoder()
        self.scene = SceneEncoder(w)
        self.chunk = 0

    def interaction_embedding(self, pos, vel_hist, allow):
        c = self.interaction(pos, vel_hist, allow)
        return c if self.use_interaction else torch.zeros_like(c)

    def forward(self, scene, focal_hm, crowd_hm, pos, vel_hist, allow):
        c = self.interaction_embedding(pos, vel_hist, allow)
        x = torch.cat([scene, focal_hm[:, None], crowd_hm[:, None]], 1)
        A = x.shape[0]
        if self.chunk <= 0 or A <= self.chunk:
            b, skips = self.scene(x, c)
        else:
            parts = [self.scene(x[i:i + self.chunk], c[i:i + self.chunk]) for i in range(0, A, self.chunk)]
            b = torch.cat([p[0] for p in parts], 0)
            skips = tuple(torch.cat([p[1][j] for p in parts], 0) for j in range(4))
        return Features(b, skips, c)
