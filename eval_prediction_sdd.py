"""SDD prediction on the standard reference split (sdd_test.pkl); data/sdd_ref_scenes.json gives each window's scene.

    python eval_prediction_sdd.py --checkpoint checkpoints/sdd_prediction.pth --ref sdd_test.pkl --rot 0 15 -15 30 -30 --flip
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from ctrl_cwm import cli
from ctrl_cwm import config as C
from ctrl_cwm.data.grid import grid2meter, heatmaps, meter2grid, scene_grid
from ctrl_cwm.data.prediction import render_ctx
from ctrl_cwm.evaluation.prediction import best_of_k, goal_probability, parse_transforms, ttst_goals, vanilla_goals
from ctrl_cwm.models import load_checkpoint


@torch.no_grad()
def run(model, ref, by_scene, sampler, transforms, batch):
    s_ade = s_fde = n = 0.
    for s, idxs in by_scene.items():
        sg = scene_grid(s)[0]
        for b0 in range(0, len(idxs), batch):
            chunk = idxs[b0:b0 + batch]
            A = len(chunk)
            ego = torch.tensor(meter2grid(np.stack([ref[i][0] for i in chunk]).astype(np.float32), s),
                               dtype=torch.float32, device=C.DEV)
            gt = torch.tensor(np.stack([ref[i][1] for i in chunk]), dtype=torch.float32, device=C.DEV)
            focal, _, pos, vel = render_ctx(ego, C.OBS - 1, torch.tensor([[j, j + 1] for j in range(A)]))
            crowd = torch.zeros_like(focal)
            for j, i in enumerate(chunk):
                nb = np.asarray(ref[i][2], np.float32)
                if nb.shape[1]:
                    ng = torch.tensor(meter2grid(nb[C.OBS - 1], s), dtype=torch.float32, device=C.DEV)
                    crowd[j] = heatmaps(ng, C.SIG).sum(0).clamp(0, 1)
            allow = torch.full((A, A), float("-inf"), device=C.DEV).fill_diagonal_(0.0)
            scene = sg[None].expand(A, -1, -1, -1)
            f = model.features(scene, focal, crowd, pos, vel, allow)
            goals = sampler(goal_probability(model, f, scene, focal, crowd, pos, vel, allow, transforms))
            ade, fde = best_of_k(model, f, goals, gt, lambda g: torch.tensor(
                grid2meter(g.cpu().numpy(), s), dtype=torch.float32, device=C.DEV))
            s_ade += ade.sum().item(); s_fde += fde.sum().item(); n += A
    return s_ade / n, s_fde / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--scenes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sdd_ref_scenes.json"))
    p.add_argument("--data-root", default=None)
    p.add_argument("--rot", type=float, nargs="*", default=[])
    p.add_argument("--flip", action="store_true")
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    a = cli.parse(p)
    model, _, meta = load_checkpoint(a.checkpoint)
    C.configure(grid=meta["grid"], sig=meta["sig"], data_root=a.data_root)
    cli.seed_everything(a.seed)
    ref, scenes = pickle.load(open(a.ref, "rb")), json.load(open(a.scenes))
    by_scene = {}
    for i, s in enumerate(scenes):
        if s:
            by_scene.setdefault(s, []).append(i)
    transforms = parse_transforms(a.rot, a.flip)
    for name, sampler in (("vanilla", vanilla_goals), ("ttst", lambda pr: ttst_goals(pr, n=a.n))):
        ade, fde = run(model.eval(), ref, by_scene, sampler, transforms, a.batch)
        print(f"{name} minADE20/minFDE20 = {ade:.4f} / {fde:.4f}", flush=True)


if __name__ == "__main__":
    main()
