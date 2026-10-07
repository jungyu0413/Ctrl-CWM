import glob
import json
import os

import numpy as np
import torch
from utils.homography import image2world

from .. import config as C
from . import maps as MU
from .grid import meter2grid, scene_grid

WAYPOINT = 4
MIN_AGENTS = 2
AUG = ("_hflip", "_vflip", "_tp", "_rev")


def dataset_stride(ds, default=5):
    for p in sorted(glob.glob(os.path.join(ds.dataset_path, "information", "*_info.json"))):
        info = json.load(open(p))
        if "fps" in info and "sim_fps" in info:
            return max(1, int(info["fps"]) // int(info["sim_fps"]))
    return default


def image_to_grid(xy, H, scene):
    if MU.is_metric(scene):
        return meter2grid(image2world(xy, H), scene)
    _, H0, W0 = scene_grid(scene)
    return np.asarray(xy, np.float64) * np.array([C.GRID / W0, C.GRID / H0])


def load_scenes(ds, stride=None):
    """CrowdES dataset -> {scene: {agents: {id: (frames, grid_xy)}, sg, fmin, fmax, stride}} for base scenes."""
    stride = stride or dataset_stride(ds)
    scenes = {}
    for s in [s for s in ds.scene_list if not any(a in s for a in AUG)]:
        item = ds[ds.scene_list.index(s)]
        df, H = item["trajectory_dense"], item["H"]
        agents = {}
        for aid, g in df.groupby("agent_id"):
            g = g.sort_values("frame")
            agents[int(aid)] = (g["frame"].to_numpy().astype(np.int64),
                                image_to_grid(g[["x", "y"]].to_numpy().astype(np.float64), H, s).astype(np.float32))
        scenes[s] = dict(agents=agents, sg=scene_grid(s)[0], fmin=int(df["frame"].min()),
                         fmax=int(df["frame"].max()), stride=stride)
    return scenes


def _pos_at(fr, grid, frame):
    i = np.searchsorted(fr, frame)
    return grid[i] if i < len(fr) and fr[i] == frame else None


def sample_window(sd, rng, horizon):
    """Agents present at a random anchor frame: history [A, OBS, 2], future [A, horizon, 2], alive mask."""
    S = sd["stride"]
    lo, hi = sd["fmin"] + (C.OBS - 1) * S, sd["fmax"] - 1
    if hi <= lo:
        return None
    f0 = int(rng.integers(lo, hi + 1))
    hist_l, fut_l, mask_l = [], [], []
    for fr, grid in sd["agents"].values():
        p0 = _pos_at(fr, grid, f0)
        if p0 is None:
            continue
        hist = np.zeros((C.OBS, 2), np.float32)
        for k in range(C.OBS):
            hf = f0 - (C.OBS - 1 - k) * S
            p = _pos_at(fr, grid, max(hf, fr[0]))
            hist[k] = p if p is not None else p0
        fut = np.zeros((horizon, 2), np.float32)
        mask = np.zeros(horizon, np.float32)
        last = p0
        for k in range(horizon):
            p = _pos_at(fr, grid, f0 + (k + 1) * S)
            if p is not None:
                last, mask[k] = p, 1.0
            fut[k] = last
        hist_l.append(hist); fut_l.append(fut); mask_l.append(mask)
    if len(hist_l) < MIN_AGENTS:
        return None
    return tuple(torch.from_numpy(np.stack(x)).to(C.DEV) for x in (hist_l, fut_l, mask_l))


def make_windows(scenes, rng, n, horizon):
    names, out, tries = list(scenes), [], 0
    while len(out) < n and tries < n * 20:
        tries += 1
        s = names[int(rng.integers(0, len(names)))]
        w = sample_window(scenes[s], rng, horizon)
        if w is not None:
            out.append((s,) + w)
    return out


def stack_batch(scenes, batch):
    """Windows -> scene [A, 9, G, G], hist, fut, mask, sse [W, 2]."""
    sg, h, f, m, sse, off = [], [], [], [], [], 0
    for s, hist, fut, mask in batch:
        n = hist.shape[0]
        sg.append(scenes[s]["sg"][None].expand(n, -1, -1, -1))
        h.append(hist); f.append(fut); m.append(mask)
        sse.append((off, off + n)); off += n
    return torch.cat(sg), torch.cat(h), torch.cat(f), torch.cat(m), torch.tensor(sse, device=C.DEV)


def last_alive(mask):
    li = (mask.shape[1] - 1) - torch.argmax(torch.flip(mask, [1]), dim=1)
    return torch.where(mask.sum(1) > 0, li, torch.zeros_like(li))


def waypoint_goal(k, fut, li):
    """Training goal at rollout step k: the ground-truth position WAYPOINT steps ahead."""
    return fut[torch.arange(fut.shape[0], device=fut.device), torch.clamp(li, max=k + WAYPOINT)]
