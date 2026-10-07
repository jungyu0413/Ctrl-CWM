"""Crowd generation and run-time control in the CrowdES protocol (Sec. 3.4).

    python generate.py --config configs/generation.yaml --dataset zara1 --checkpoint checkpoints/zara1_ctrlcwm.pth
    python generate.py --config configs/generation.yaml --dataset zara1 --checkpoint ... \\
        --avoid 83,123,25 --w-user 25 --tag avoid
"""
import argparse
import json
import os

import numpy as np
import torch

from ctrl_cwm import cli
from ctrl_cwm import config as C
from ctrl_cwm.models import load_checkpoint
from ctrl_cwm.planning import CEMPlanner, PlannerConfig, UserCost
from ctrl_cwm.simulation import crowdes


def average(results):
    acc = {}
    for r in results:
        for grp, vals in r.items():
            for k, v in vals.items():
                acc.setdefault(grp, {}).setdefault(k, []).append(float(v))
    return {g: {k: float(np.mean(v)) for k, v in kv.items()} for g, kv in acc.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--crowdes-root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="output")
    p.add_argument("--tag", default="ctrlcwm")
    p.add_argument("--emitter", choices=["diffusion", "surface"], default="diffusion")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--planner", choices=["critic", "user", "none"], default="critic")
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--avoid", type=cli.floats, action="append", default=[], help="cx,cy,r in grid px")
    p.add_argument("--attract", type=cli.floats, default=None, help="cx,cy in grid px")
    p.add_argument("--w-user", type=float, default=0.0)
    p.add_argument("--control-onset", type=float, default=1 / 3, help="fraction of the episode before control starts")
    p.add_argument("--seed-offset", type=int, default=0)
    a = cli.parse(p)
    a.avoid = [cli.floats(z) if isinstance(z, str) else tuple(z) for z in a.avoid]

    ck = os.path.abspath(a.checkpoint)
    out = os.path.abspath(a.out)
    model, critic, meta = load_checkpoint(ck)
    C.configure(grid=meta["grid"], sig=meta["sig"], data_root=a.data_root)
    crowdes.setup(a.crowdes_root)
    from ctrl_cwm.simulation.simulator import CtrlCWMSimulator
    from utils.metrics import compute_metrics

    user = UserCost(avoid=a.avoid, attract=a.attract) if (a.avoid or a.attract) else None
    planner = CEMPlanner(PlannerConfig(a.planner, a.candidates, a.iters, a.horizon, a.sigma, w_user=a.w_user),
                         critic.eval() if critic is not None else None, user)
    cfg, ds = crowdes.load_dataset(a.dataset, "test")
    cfg.crowd_emitter.type = "CrowdES" if a.emitter == "diffusion" else "surface"
    sim = CtrlCWMSimulator(cfg, model, planner, control_onset=a.control_onset if user else 0.0)

    gen_dir = os.path.join(out, "generated", a.dataset)
    os.makedirs(gen_dir, exist_ok=True)
    results = {}
    for i, scene in enumerate(ds.scene_list):
        data = ds[i]
        gt = data["trajectory_dense"].copy()
        gt["scene"] = scene
        sim.scene_stem = scene
        results[scene] = []
        for t in range(a.trials):
            sim.initialize_scene(data["img"], data["seg"], data["walkable"], data["navmesh"], data["H"])
            with torch.no_grad():
                gen = sim.generate(a.max_frames or data["size"]["length"], seed=a.seed_offset + t)
            gen["scene"] = scene
            gen.to_pickle(os.path.join(gen_dir, f"{scene}-{a.tag}-trial{t}.pkl.gz"))
            results[scene].append(compute_metrics(gen, gt, data["size"], data["H"]))
            print(f"[{a.dataset}-{a.tag}] {scene} trial {t + 1}/{a.trials}", flush=True)

    summary = {s: average(r) for s, r in results.items()}
    summary["mean"] = average([m for r in results.values() for m in r])
    with open(os.path.join(out, f"{a.dataset}_{a.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for grp, vals in summary["mean"].items():
        print(f"  {grp}: " + "  ".join(f"{k}={v:.4f}" for k, v in vals.items()), flush=True)


if __name__ == "__main__":
    main()
