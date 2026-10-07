"""Run CrowdES's released checkpoint through its own test protocol.

    python tools/run_crowdes_baseline.py zara1 --crowdes-root /path/to/Crowd-Behavior-Generation
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--crowdes-root", default=None)
    a = p.parse_args()

    from ctrl_cwm.simulation import crowdes
    crowdes.setup(a.crowdes_root)
    _orig = argparse.ArgumentParser.add_argument

    def _add_argument(self, *args, **kw):
        if args and args[0] == "--model_config":
            kw["type"] = str
        return _orig(self, *args, **kw)

    argparse.ArgumentParser.add_argument = _add_argument
    sys.argv = ["trainval.py", "--model_config", f"./configs/model/CrowdES_{a.dataset}.yaml",
                "--model_train", "simulator", "--test"]
    exec(open("trainval.py").read(), {"__name__": "__main__", "__file__": "trainval.py"})


if __name__ == "__main__":
    main()
