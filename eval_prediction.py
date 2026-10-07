"""Prediction heads: best-of-20 minADE / minFDE (Appendix E).

    python eval_prediction.py --dataset zara1 --checkpoint checkpoints/zara1_prediction.pth --rot 0 15 -15 30 -30 --flip
"""
import argparse

import torch

from ctrl_cwm import cli
from ctrl_cwm import config as C
from ctrl_cwm.data.prediction import build, grid_to_native, test_loader
from ctrl_cwm.evaluation.prediction import best_of_k, goal_probability, parse_transforms, ttst_goals, vanilla_goals
from ctrl_cwm.models import load_checkpoint


@torch.no_grad()
def run(model, loader, sampler, transforms):
    s_ade = s_fde = n = 0.
    for b in loader:
        scene, focal, crowd, _, pos, vel, allow, traj, sse, scenes = build(b)
        f = model.features(scene, focal, crowd, pos, vel, allow)
        goals = sampler(goal_probability(model, f, scene, focal, crowd, pos, vel, allow, transforms))
        ade, fde = best_of_k(model, f, goals, traj[:, C.OBS:C.T], lambda g: grid_to_native(g, sse, scenes))
        s_ade += ade.sum().item(); s_fde += fde.sum().item(); n += ade.shape[0]
    return s_ade / n, s_fde / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--rot", type=float, nargs="*", default=[])
    p.add_argument("--flip", action="store_true")
    p.add_argument("--sampler", choices=["ttst", "vanilla"], default="ttst")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--skip-test", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    a = cli.parse(p)
    model, _, meta = load_checkpoint(a.checkpoint)
    C.configure(grid=meta["grid"], sig=meta["sig"], data_root=a.data_root)
    cli.seed_everything(a.seed)
    sampler = (lambda pr: ttst_goals(pr, n=a.n)) if a.sampler == "ttst" else vanilla_goals
    ade, fde = run(model.eval(), test_loader(a.dataset, a.batch, a.skip_test), sampler, parse_transforms(a.rot, a.flip))
    print(f"[{a.dataset}] minADE20/minFDE20 = {ade:.4f} / {fde:.4f}", flush=True)


if __name__ == "__main__":
    main()
