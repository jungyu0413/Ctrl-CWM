"""Control compliance (Eq. 13) from paired commanded / free runs over the post-activation window.

    python tools/score_control.py --gen-dir output/generated/zara1 --scene crowds_zara01 --cmd-tag avoid \\
        --free-tag ctrlcwm --region 83,123,25 --mode avoid --to-grid 576,720,192
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def occupancy(df, region, to_grid, onset):
    df = df[df["frame"] >= onset * (df["frame"].max() + 1)]
    H, W, G = to_grid
    xy = df[["x", "y"]].to_numpy() * np.array([G / W, G / H])
    return float((np.linalg.norm(xy - np.array(region[:2]), axis=-1) <= region[2]).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-dir", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--cmd-tag", required=True)
    p.add_argument("--free-tag", required=True)
    p.add_argument("--region", required=True, help="cx,cy,r in grid px")
    p.add_argument("--mode", choices=["avoid", "attract"], default="avoid")
    p.add_argument("--to-grid", required=True, help="H,W,GRID")
    p.add_argument("--onset", type=float, default=1 / 3)
    a = p.parse_args()
    region = tuple(float(x) for x in a.region.split(","))
    to_grid = tuple(float(x) for x in a.to_grid.split(","))

    def runs(tag):
        return {int(f.split("trial")[-1].split(".")[0]): f
                for f in glob.glob(os.path.join(a.gen_dir, f"{a.scene}-{tag}-trial*.pkl.gz"))}

    cmd, free = runs(a.cmd_tag), runs(a.free_tag)
    scores = []
    for t in sorted(set(cmd) & set(free)):
        occ_cmd = occupancy(pd.read_pickle(cmd[t]), region, to_grid, a.onset)
        occ_free = occupancy(pd.read_pickle(free[t]), region, to_grid, a.onset)
        if a.mode == "avoid" and occ_free > 0:
            scores.append(1 - occ_cmd / occ_free)
        elif a.mode == "attract" and occ_free < 1:
            scores.append((occ_cmd - occ_free) / (1 - occ_free))
    print(f"{a.mode} compliance = {np.mean(scores):.4f} over {len(scores)} paired trials")


if __name__ == "__main__":
    main()
