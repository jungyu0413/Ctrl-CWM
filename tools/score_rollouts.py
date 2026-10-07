"""Re-score saved rollouts with CrowdES's eight metrics (independent of the run that produced them).

    python tools/score_rollouts.py --dir output/generated/zara1 --tag ctrlcwm --dataset zara1
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="directory of <scene>-<tag>-trial<t>.pkl.gz files")
    p.add_argument("--tag", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--crowdes-root", default=None)
    p.add_argument("--trials", type=int, default=0, help="0 = all found")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    rolls = sorted(glob.glob(os.path.join(os.path.abspath(a.dir), f"*-{a.tag}-trial*.pkl.gz")))
    if not rolls:
        raise SystemExit(f"no rollouts matching *-{a.tag}-trial*.pkl.gz in {a.dir}")
    out_json = os.path.abspath(a.json) if a.json else None

    from ctrl_cwm.simulation import crowdes
    crowdes.setup(a.crowdes_root)
    import pandas as pd
    from utils.metrics import compute_metrics

    _, ds = crowdes.load_dataset(a.dataset, "test")
    index = {s: i for i, s in enumerate(ds.scene_list)}
    cols = ["scene", "agent_id", "agent_type", "frame", "x", "y"]
    per = []
    for path in rolls:
        scene = os.path.basename(path).split("-")[0]
        if scene not in index:
            print(f"  skip {os.path.basename(path)}: scene not in the test split")
            continue
        d = ds[index[scene]]
        gen = pd.read_pickle(path)
        gen = gen.assign(scene=scene) if "scene" not in gen else gen
        gen = gen.assign(agent_type=0) if "agent_type" not in gen else gen
        gt = d["trajectory_dense"].copy()
        if "scene" not in gt.columns:
            gt["scene"] = scene
        per.append(compute_metrics(gen[cols], gt[cols], d["size"], d["H"]))
        if a.trials and len(per) >= a.trials:
            break

    acc = {}
    for m in per:
        for grp, vals in m.items():
            for k, v in vals.items():
                acc.setdefault(grp, {}).setdefault(k, []).append(float(v))
    mean = {g: {k: sum(v) / len(v) for k, v in kv.items()} for g, kv in acc.items()}
    print(f"=== {a.tag} ({len(per)} rollouts) ===")
    for grp, vals in mean.items():
        for k, v in vals.items():
            print(f"  {grp[:5]} {k:<22} {v:.6f}")
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        json.dump({"mean": mean, "n": len(per), "tag": a.tag}, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
