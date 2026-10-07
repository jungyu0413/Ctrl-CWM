import os

import torch

OBS, PRED = 8, 12
T = OBS + PRED
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 192
SIG = 5.0
DATA_ROOT = os.environ.get("CTRL_CWM_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))


def configure(grid=None, sig=None, data_root=None):
    global GRID, SIG, DATA_ROOT
    GRID = int(grid) if grid is not None else GRID
    SIG = float(sig) if sig is not None else SIG
    DATA_ROOT = os.path.abspath(data_root) if data_root is not None else DATA_ROOT


def maps_dir():
    return os.path.join(DATA_ROOT, "maps")


def datasets_dir():
    return os.path.join(DATA_ROOT, "datasets")
