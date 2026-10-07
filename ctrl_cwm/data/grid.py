import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from .. import config as C
from . import maps as MU

_cache = {}


def scene_grid(scene):
    """-> ([9, GRID, GRID] walkability + semantic channels, native H, native W)."""
    key = (scene, C.GRID)
    if key not in _cache:
        walk = MU.load_walkable(scene).astype(np.float32)
        chan = np.concatenate([walk[None], MU.seg_to_onehot(MU.load_seg(scene))], 0)
        t = torch.from_numpy(chan).to(C.DEV)
        t = F.interpolate(t[None], size=(C.GRID, C.GRID), mode="bilinear", align_corners=False)[0]
        _cache[key] = (t, *walk.shape)
    return _cache[key]


def meter2grid(coords, scene):
    _, H, W = scene_grid(scene)
    px = MU.to_map_pixels(np.asarray(coords), scene)
    g = px.copy()
    g[..., 0] = px[..., 0] * (C.GRID / W)
    g[..., 1] = px[..., 1] * (C.GRID / H)
    return g


def grid2meter(g, scene):
    _, H, W = scene_grid(scene)
    px = np.asarray(g, dtype=float).copy()
    px[..., 0] = px[..., 0] * (W / C.GRID)
    px[..., 1] = px[..., 1] * (H / C.GRID)
    if not MU.is_metric(scene):
        return px
    c = px.reshape(-1, 2)
    m = (MU.homography(scene) @ np.stack([c[:, 0], c[:, 1], np.ones_like(c[:, 0])], -1).T).T
    return (m / m[:, [2]])[:, :2].reshape(px.shape)


def heatmaps(grid_xy, sigma, grid=None):
    grid = C.GRID if grid is None else grid
    ys = torch.arange(grid, device=grid_xy.device).float()
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    d2 = (xx - grid_xy[..., 0, None, None]) ** 2 + (yy - grid_xy[..., 1, None, None]) ** 2
    return torch.exp(-d2 / (2 * sigma ** 2))


def soft_argmax(hm):
    G = hm.shape[-1]
    p = F.softmax(hm.reshape(*hm.shape[:-2], G * G), -1).reshape(*hm.shape)
    xs = torch.arange(G, device=hm.device).float()
    return torch.stack([(p.sum(-2) * xs).sum(-1), (p.sum(-1) * xs).sum(-1)], -1)


def index_to_xy(idx):
    return torch.stack([(idx % C.GRID).float(), (idx // C.GRID).float()], -1)


def rand_affine_grid_coords(scene_img, coords, scale_rng=(0.7, 1.35)):
    """Same random rotation, flip and scale for a scene raster [C, G, G] and grid coordinates [..., 2]."""
    G = scene_img.shape[-1]
    th = random.uniform(-math.pi, math.pi)
    fl = -1.0 if random.random() < 0.5 else 1.0
    sc = random.uniform(*scale_rng)
    c, s = math.cos(th), math.sin(th)
    A = sc * torch.tensor([[c * fl, -s], [s * fl, c]], device=scene_img.device, dtype=torch.float32)
    aug_coords = ((coords / (G - 1) * 2 - 1) @ A.T + 1) / 2 * (G - 1)
    theta = torch.zeros(1, 2, 3, device=scene_img.device)
    theta[0, :, :2] = torch.linalg.inv(A)
    grid = F.affine_grid(theta, (1, scene_img.shape[0], G, G), align_corners=False)
    return F.grid_sample(scene_img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0], aug_coords
