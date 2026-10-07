"""Stage 1: learn the world model from trajectory prediction (Sec. 3.2).

    python train_prediction.py --config configs/prediction.yaml --dataset zara1 --width 0.25
"""
import argparse
import os
import time

import torch

from ctrl_cwm import cli
from ctrl_cwm import config as C
from ctrl_cwm.data.prediction import loaders
from ctrl_cwm.models import CtrlCWM, save_checkpoint
from ctrl_cwm.training.stage1 import evaluate, train_epoch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--no-interaction", action="store_true")
    p.add_argument("--grid", type=int, default=192)
    p.add_argument("--sig", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w-way", type=float, default=1.0)
    p.add_argument("--w-dyn", type=float, default=0.5)
    p.add_argument("--skip", type=int, default=1)
    p.add_argument("--skip-test", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = cli.parse(p)
    C.configure(grid=a.grid, sig=a.sig, data_root=a.data_root)
    cli.seed_everything(a.seed)

    tr, te = loaders(a.dataset, a.batch, skip=a.skip, skip_test=a.skip_test)
    model = CtrlCWM(width=a.width, use_interaction=not a.no_interaction).to(C.DEV)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.dataset}_prediction.pth")
    best = float("inf")
    for epoch in range(a.epochs):
        t0 = time.time()
        goal, way, dyn = train_epoch(model, tr, opt, a.accum, a.w_way, a.w_dyn)
        sch.step()
        msg = f"[{a.dataset}] epoch {epoch + 1}/{a.epochs} {time.time() - t0:.0f}s goal={goal:.4f} way={way:.4f} dyn={dyn:.4f}"
        if (epoch + 1) % a.eval_every == 0:
            ade, fde = evaluate(model, te)
            msg += f" ADE20={ade:.4f} FDE20={fde:.4f}"
            if ade < best:
                best = ade
                save_checkpoint(path, model)
                msg += " *"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
