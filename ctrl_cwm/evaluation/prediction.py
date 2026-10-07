import math

import torch
import torch.nn.functional as F

from .. import config as C
from ..data.grid import heatmaps, index_to_xy, soft_argmax


def parse_transforms(rot, flip):
    return [(r, False) for r in rot] + ([(r, True) for r in rot] if flip else [])


def _rot(deg, flip):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    f = -1.0 if flip else 1.0
    return torch.tensor([[c * f, -s], [s * f, c]], device=C.DEV)


def _warp(img, A):
    theta = torch.zeros(1, 2, 3, device=C.DEV)
    theta[0, :, :2] = torch.linalg.inv(A)
    grid = F.affine_grid(theta.expand(img.shape[0], 2, 3), img.shape, align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


@torch.no_grad()
def goal_probability(model, f, scene, focal, crowd, pos, vel, allow, transforms=()):
    """Goal distribution [A, G*G], averaged over rotated / flipped inputs when transforms are given."""
    A, G = scene.shape[0], C.GRID
    if not transforms:
        return F.softmax(model.goal_logits(f).reshape(A, -1), -1)
    prob = torch.zeros(A, G, G, device=C.DEV)
    for deg, flip in transforms:
        M = _rot(deg, flip)
        p = ((pos * G / (G - 1) * 2 - 1) @ M.T + 1) / 2 * (G - 1) / G
        fr = model.features(_warp(scene, M), _warp(focal[:, None], M)[:, 0], _warp(crowd[:, None], M)[:, 0],
                            p, vel @ M.T, allow)
        pr = F.softmax(model.goal_logits(fr).reshape(A, -1), -1).reshape(A, G, G)
        prob = prob + _warp(pr[:, None], torch.linalg.inv(M))[:, 0].clamp(min=0)
    prob = prob.reshape(A, -1)
    return prob / prob.sum(-1, keepdim=True).clamp(min=1e-8)


@torch.no_grad()
def vanilla_goals(p, k=20):
    return index_to_xy(torch.multinomial(p, k, replacement=True))


@torch.no_grad()
def ttst_goals(p, k=20, n=2000, iters=8):
    cand = index_to_xy(torch.multinomial(p, n, replacement=True))
    cen = cand[:, torch.randperm(n, device=C.DEV)[:k]].clone()
    for _ in range(iters):
        oh = F.one_hot(torch.cdist(cand, cen).argmin(-1), k).float()
        cen = (oh.transpose(1, 2) @ cand) / oh.sum(1).clamp(min=1)[..., None]
    return cen


@torch.no_grad()
def best_of_k(model, f, goals, gt, to_native):
    ade = torch.full((goals.shape[0],), 1e9, device=C.DEV)
    fde = torch.full_like(ade, 1e9)
    for j in range(goals.shape[1]):
        way, _ = model.waypoint_logits(f, heatmaps(goals[:, j], C.SIG))
        err = (to_native(soft_argmax(way)) - gt).norm(dim=-1)
        ade = torch.minimum(ade, err.mean(-1))
        fde = torch.minimum(fde, err[:, -1])
    return ade, fde
