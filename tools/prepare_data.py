"""Build SDD and GCS maps and 2.5 Hz prediction splits from the CrowdES preprocessed release.

    python tools/prepare_data.py sdd --crowdes-root /path/to/Crowd-Behavior-Generation
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image

AUG = ("_hflip", "_vflip", "_rev", "_tp")


def build_sdd(cb, out):
    src = os.path.join(cb, "sdd")
    for split in ("train", "test"):
        if not os.path.isdir(f"{src}/{split}/trajectory"):
            print(f"  [skip] {src}/{split} not found")
            continue
        stems = sorted(n[:-len("_trajectory.csv")] for n in os.listdir(f"{src}/{split}/trajectory")
                       if n.endswith("_trajectory.csv"))
        for scene in [s for s in stems if not any(a in s for a in AUG)]:
            d = os.path.join(out, "maps", "sdd", scene)
            os.makedirs(d, exist_ok=True)
            shutil.copy(f"{src}/{split}/segmentation/{scene}_seg.png", f"{d}/seg.png")
            walk = np.array(Image.open(f"{src}/{split}/navmesh/{scene}_walkable_area.png").convert("L"))
            Image.fromarray(np.where(walk > 0, 255, 0).astype(np.uint8)).save(f"{d}/walkable.png")

            df = pd.read_csv(f"{src}/{split}/trajectory/{scene}_trajectory.csv")
            df = df[df["agent_type"] == 0].copy()
            if df.empty:
                continue
            step = (df["frame"] - int(df["frame"].min())) // 6
            keep = (step % 2) == 0
            df = df.loc[keep].copy()
            df["f"] = step[keep] // 2
            os.makedirs(os.path.join(out, "datasets", "sdd", split), exist_ok=True)
            df[["f", "agent_id", "x", "y"]].sort_values(["f", "agent_id"]).to_csv(
                os.path.join(out, "datasets", "sdd", split, f"{scene}.txt"),
                sep="\t", header=False, index=False, float_format="%.4f")
            print(f"  sdd/{split}/{scene}: {len(df)} rows")


def build_gcs(cb, out):
    src = os.path.join(cb, "gcs")
    d = os.path.join(out, "maps", "gcs", "terminal")
    os.makedirs(d, exist_ok=True)
    seg_src = f"{src}/test/segmentation/terminal_seg.png"
    shutil.copy(seg_src, f"{d}/seg.png")
    seg = np.array(Image.open(seg_src).convert("L"))
    Image.fromarray(np.where(np.isin(seg, [1, 2, 3, 4, 5]), 0, 255).astype(np.uint8)).save(f"{d}/walkable.png")

    def dense(split):
        dd = f"{src}/{split}/trajectory_dense"
        fn = [f for f in os.listdir(dd) if f.endswith(".csv") and not any(a in f for a in AUG)]
        assert len(fn) == 1, fn
        return pd.read_csv(os.path.join(dd, fn[0]))

    def write(df, split):
        f0 = int(df["frame"].min())
        o = df.loc[((df["frame"] - f0) % 10) == 0].copy()
        o["f"] = (o["frame"] - f0) // 10
        os.makedirs(os.path.join(out, "datasets", "gcs25", split), exist_ok=True)
        o[["f", "agent_id", "x", "y"]].sort_values(["f", "agent_id"]).to_csv(
            os.path.join(out, "datasets", "gcs25", split, "terminal.txt"),
            sep="\t", header=False, index=False, float_format="%.4f")
        print(f"  gcs25/{split}: {len(o)} rows")

    te, tr = dense("test"), dense("train")
    write(te, "test")
    cut = tr["frame"].quantile(0.85)
    write(tr[tr["frame"] <= cut], "train")
    write(tr[tr["frame"] > cut], "val")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["sdd", "gcs"])
    p.add_argument("--crowdes-root", default=os.environ.get("CROWDES_ROOT"), required=not os.environ.get("CROWDES_ROOT"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    a = p.parse_args()
    cb = os.path.join(a.crowdes_root, "datasets", "preprocessed")
    (build_sdd if a.dataset == "sdd" else build_gcs)(cb, a.out)


if __name__ == "__main__":
    main()
