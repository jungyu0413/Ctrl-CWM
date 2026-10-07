import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .. import config as C
from .grid import grid2meter, heatmaps, index_to_xy, meter2grid, rand_affine_grid_coords, scene_grid


def read_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            v = line.split()
            if len(v) >= 4:
                rows.append([float(x) for x in v[:4]])
    return np.asarray(rows)


class TrajectoryDataset(Dataset):
    """Windows of OBS + PRED frames holding every agent present for the whole window (at least two)."""

    def __init__(self, data_dir, obs_len=C.OBS, pred_len=C.PRED, skip=1):
        super().__init__()
        seq_len = obs_len + pred_len
        seqs, counts, scenes = [], [], []
        for name in sorted(os.listdir(data_dir)):
            data = read_file(os.path.join(data_dir, name))
            if data.size == 0:
                continue
            data = data[np.argsort(data[:, 0], kind="stable")]
            frames, starts = np.unique(data[:, 0], return_index=True)
            frames = frames.tolist()
            frame_data = np.split(data, starts[1:])
            num_seq = int(math.ceil((len(frames) - seq_len + 1) / skip))
            for idx in range(0, max(num_seq, 0) * skip + 1, skip):
                window = frame_data[idx:idx + seq_len]
                if len(window) < seq_len:
                    break
                cur = np.concatenate(window, axis=0)
                seq = []
                for ped in np.unique(cur[:, 1]):
                    ps = np.around(cur[cur[:, 1] == ped], decimals=4)
                    pf = frames.index(ps[0, 0]) - idx
                    if pf == 0 and ps.shape[0] == seq_len and frames.index(ps[-1, 0]) - idx + 1 == seq_len:
                        seq.append(ps[:, 2:].T)
                if len(seq) > 1:
                    seqs.append(np.stack(seq)); counts.append(len(seq)); scenes.append(os.path.splitext(name)[0])
        seqs = np.concatenate(seqs, axis=0)
        self.obs_traj = torch.from_numpy(seqs[:, :, :obs_len]).float()
        self.pred_traj = torch.from_numpy(seqs[:, :, obs_len:]).float()
        cum = [0] + np.cumsum(counts).tolist()
        self.seq_start_end = list(zip(cum, cum[1:]))
        self.scene = scenes

    def __len__(self):
        return len(self.seq_start_end)

    def __getitem__(self, index):
        s, e = self.seq_start_end[index]
        return self.obs_traj[s:e], self.pred_traj[s:e], self.scene[index]


def collate(batch):
    obs, pred, scene = zip(*batch)
    cum = [0] + np.cumsum([t.shape[0] for t in obs]).tolist()
    return {"obs_traj": torch.cat(obs, 0), "pred_traj": torch.cat(pred, 0),
            "seq_start_end": torch.tensor(list(zip(cum, cum[1:])), dtype=torch.long), "scene": list(scene)}


def test_loader(dataset, bs, skip_test=1, workers=4):
    root = os.path.join(C.datasets_dir(), dataset)
    return DataLoader(TrajectoryDataset(f"{root}/test", skip=skip_test), bs, shuffle=False,
                      collate_fn=collate, num_workers=workers)


def loaders(dataset, bs, skip=1, skip_test=1, workers=4):
    root = os.path.join(C.datasets_dir(), dataset)
    tr = DataLoader(TrajectoryDataset(f"{root}/train", skip=skip), bs, shuffle=True,
                    collate_fn=collate, num_workers=workers)
    return tr, test_loader(dataset, bs, skip_test, workers)


def allow_mask(sse, A):
    m = torch.full((A, A), float("-inf"), device=C.DEV)
    for s, e in sse:
        m[s:e, s:e] = 0.0
    return m


def render_ctx(pos_hist, t, sse):
    """Model inputs at step t of trajectories pos_hist [A, T, 2] (grid px) grouped into scenes by sse:
    focal heatmap (last OBS positions), crowd heatmap (co-present agents at t), position / G, velocities / G."""
    win = pos_hist[:, max(0, t - C.OBS + 1):t + 1]
    focal = heatmaps(win, C.SIG).sum(1).clamp(0, 1)
    pt = pos_hist[:, t]
    last = heatmaps(pt, C.SIG)
    crowd = torch.zeros_like(focal)
    for s, e in sse.tolist():
        crowd[s:e] = (last[s:e].sum(0, keepdim=True) - last[s:e]).clamp(0, 1)
    if win.shape[1] >= 2:
        v = win[:, 1:] - win[:, :-1]
        v = torch.cat([torch.zeros_like(v[:, :1]), v], 1) / C.GRID
    else:
        v = torch.zeros(pos_hist.shape[0], 1, 2, device=pos_hist.device)
    return focal, crowd, pt / C.GRID, v


def build(b, train=False):
    """Batch -> (scene, focal_hm, crowd_hm, grid trajectories, pos, vel_hist, allow, native trajectories, sse, scenes)."""
    traj = torch.cat([b["obs_traj"].permute(0, 2, 1), b["pred_traj"].permute(0, 2, 1)], 1).to(C.DEV)
    sse, scenes = b["seq_start_end"], b["scene"]
    scene_l, grid_l = [], []
    for wi, (s, e) in enumerate(sse.tolist()):
        sg = scene_grid(scenes[wi])[0]
        g = torch.tensor(meter2grid(traj[s:e].cpu().numpy(), scenes[wi]), dtype=torch.float32, device=C.DEV)
        if train:
            sg, g = rand_affine_grid_coords(sg, g)
        scene_l.append(sg[None].expand(e - s, -1, -1, -1))
        grid_l.append(g)
    gridtraj = torch.cat(grid_l, 0)
    focal, crowd, pos, vel = render_ctx(gridtraj, C.OBS - 1, sse)
    return torch.cat(scene_l, 0), focal, crowd, gridtraj, pos, vel, allow_mask(sse, traj.shape[0]), traj, sse, scenes


@torch.no_grad()
def sample_goals(goal_logits, k):
    p = torch.softmax(goal_logits.reshape(goal_logits.shape[0], -1), -1)
    return index_to_xy(torch.multinomial(p, k, replacement=True))


def grid_to_native(pg, sse, scenes):
    pgn = pg.detach().cpu().numpy()
    out = np.empty_like(pgn)
    for wi, (s, e) in enumerate(sse.tolist()):
        out[s:e] = grid2meter(pgn[s:e], scenes[wi])
    return torch.tensor(out, device=C.DEV)
