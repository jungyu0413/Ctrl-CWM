import torch
import torch.nn.functional as F

from .. import config as C
from ..data.grid import heatmaps, soft_argmax


def heatmap_loss(logits, target_xy, pos_weight):
    """BCE to Gaussian targets + distance of the soft-argmax coordinate (Eq. 3 terms)."""
    bce = F.binary_cross_entropy_with_logits(logits, heatmaps(target_xy, C.SIG), pos_weight=pos_weight)
    return bce + (soft_argmax(logits) - target_xy).norm(dim=-1).mean() / C.GRID


def soft_dtw(x, y, gamma=1.0):
    """x, y [A, T, 2] -> [A]."""
    A, T, _ = x.shape
    D = torch.cdist(x, y) / C.GRID
    inf = torch.full((A,), 1e6, device=x.device)
    prev = [torch.zeros(A, device=x.device)] + [inf] * T
    for i in range(1, T + 1):
        cur = [inf]
        for j in range(1, T + 1):
            st = torch.stack([prev[j], cur[j - 1], prev[j - 1]], -1)
            cur.append(D[:, i - 1, j - 1] - gamma * torch.logsumexp(-st / gamma, -1))
        prev = cur
    return prev[T]


def imitation_loss(traj, gt, w_pos, w_dtw, gamma):
    """Eq. 4, per agent: traj, gt [A, L, 2] -> [A]."""
    return w_pos * (traj - gt).norm(dim=-1).mean(1) / C.GRID + w_dtw * soft_dtw(traj, gt, gamma)


def kinematic_loss(traj, gt):
    """Eq. 5, per agent: speed and speed-change mismatch."""
    v = (traj[:, 1:] - traj[:, :-1]).norm(dim=-1) / C.GRID
    v_gt = (gt[:, 1:] - gt[:, :-1]).norm(dim=-1) / C.GRID
    return (v - v_gt).abs().mean(1) + ((v[:, 1:] - v[:, :-1]) - (v_gt[:, 1:] - v_gt[:, :-1])).abs().mean(1)


def collision_count(traj, focal, radius):
    """Eq. 8: neighbours of agent `focal` within radius, averaged over steps. traj [W, A, L, 2] -> [W]."""
    d = (traj - traj[:, focal:focal + 1]).norm(dim=-1)
    d[:, focal] = float("inf")
    return (d < radius).float().sum(1).mean(-1)
