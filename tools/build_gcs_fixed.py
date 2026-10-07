"""Create `gcs_fixed` in the CrowdES repository: GCS test split with frames rebased and length corrected.

    python tools/build_gcs_fixed.py --crowdes-root /path/to/Crowd-Behavior-Generation
"""
import argparse
import json
import os
import shutil

import pandas as pd

SPLIT_FRAME = 96001


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crowdes-root", default=os.environ.get("CROWDES_ROOT"), required=not os.environ.get("CROWDES_ROOT"))
    root = os.path.abspath(p.parse_args().crowdes_root)
    pre = os.path.join(root, "datasets", "preprocessed")
    src, dst = f"{pre}/gcs", f"{pre}/gcs_fixed"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(f"{dst}/test")
    os.symlink(f"{src}/train", f"{dst}/train")
    for d in os.listdir(f"{src}/test"):
        if d not in ("trajectory", "trajectory_dense", "information"):
            os.symlink(f"{src}/test/{d}", f"{dst}/test/{d}")
    for sub in ("trajectory", "trajectory_dense", "information"):
        os.makedirs(f"{dst}/test/{sub}")
    for sub in ("trajectory", "trajectory_dense"):
        fn = f"terminal_{sub}.csv"
        df = pd.read_csv(f"{src}/test/{sub}/{fn}")
        df["frame"] = df["frame"] - SPLIT_FRAME
        assert df["frame"].min() >= 0
        df.to_csv(f"{dst}/test/{sub}/{fn}", index=False)

    info = json.load(open(f"{src}/test/information/terminal_info.json"))
    dense = pd.read_csv(f"{dst}/test/trajectory_dense/terminal_trajectory_dense.csv")
    length = int(dense["frame"].max()) + 1
    count = dense.groupby("frame")["agent_id"].nunique().reindex(range(length), fill_value=0)
    info["length"] = length
    info["population_probability"] = {str(k): float(v) for k, v in (count.value_counts().sort_index() / length).items()}
    json.dump(info, open(f"{dst}/test/information/terminal_info.json", "w"))

    cfg = os.path.join(root, "configs")
    with open(f"{cfg}/dataset/gcs.yaml") as f:
        text = f.read()
    text = (text.replace("dataset_name: gcs", "dataset_name: gcs_fixed")
                .replace("preprocessed/gcs/", "preprocessed/gcs_fixed/")
                .replace("dataset_fps: 25", "dataset_fps: 30"))
    with open(f"{cfg}/dataset/gcs_fixed.yaml", "w") as f:
        f.write(text)
    with open(f"{cfg}/model/CrowdES_gcs.yaml") as f:
        text = f.read()
    with open(f"{cfg}/model/CrowdES_gcs_fixed.yaml", "w") as f:
        f.write(text.replace("./configs/dataset/gcs.yaml", "./configs/dataset/gcs_fixed.yaml"))
    ck = os.path.join(root, "checkpoints", "gcs_fixed")
    if not os.path.lexists(ck):
        os.symlink(os.path.join(root, "checkpoints", "gcs"), ck)
    print(f"gcs_fixed: length={length}, {dense['agent_id'].nunique()} test agents -> {dst}")


if __name__ == "__main__":
    main()
