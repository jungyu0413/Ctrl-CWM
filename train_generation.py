"""Stage 2: learn behavior in imagination, actor and critic on the frozen world model (Sec. 3.3).

    python train_generation.py --config configs/behavior.yaml --dataset zara1 \\
        --world-model checkpoints/zara1_prediction.pth
"""
import argparse
import dataclasses
import os

import numpy as np

from ctrl_cwm import cli
from ctrl_cwm import config as C
from ctrl_cwm.models import CtrlCWM, load_checkpoint
from ctrl_cwm.simulation import crowdes
from ctrl_cwm.training.stage2 import BehaviorConfig, BehaviorLearner


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--world-model", default=None, help="stage-1 checkpoint (omit for a randomly initialized encoder)")
    p.add_argument("--width", type=float, default=0.25)
    p.add_argument("--crowdes-root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--grid", type=int, default=192)
    p.add_argument("--sig", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    for f in dataclasses.fields(BehaviorConfig):
        name = "--" + f.name.replace("_", "-")
        if f.type is bool or isinstance(f.default, bool):
            p.add_argument(name, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(name, type=type(f.default), default=f.default)
    a = cli.parse(p)
    C.configure(grid=a.grid, sig=a.sig, data_root=a.data_root)
    out = os.path.abspath(a.out)
    world_model = os.path.abspath(a.world_model) if a.world_model else None
    crowdes.setup(a.crowdes_root, compat=False)
    from ctrl_cwm.data.simulation import load_scenes

    cli.seed_everything(a.seed)
    model = load_checkpoint(world_model)[0] if world_model else CtrlCWM(width=a.width).to(C.DEV)
    scenes = load_scenes(crowdes.load_dataset(a.dataset, "train")[1])
    cfg = BehaviorConfig(**{f.name: getattr(a, f.name) for f in dataclasses.fields(BehaviorConfig)})
    os.makedirs(out, exist_ok=True)
    BehaviorLearner(model, scenes, cfg).fit(np.random.default_rng(a.seed), os.path.join(out, f"{a.dataset}_ctrlcwm.pth"),
                                            log=lambda s: print(f"[{a.dataset}] {s}", flush=True))


if __name__ == "__main__":
    main()
