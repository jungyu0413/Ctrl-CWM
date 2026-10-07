import functools
import os

import numpy as np
from PIL import Image

from .. import config as C

FAMILIES = ["eth-ucy", "sdd", "gcs"]
NUM_SEG_CLASSES = 8
ALIAS = {"seq_eth": "biwi_eth", "seq_hotel": "biwi_hotel", "eth": "biwi_eth", "hotel": "biwi_hotel"}


def _norm(scene):
    for suf in ("_train", "_val", "_test"):
        if scene.endswith(suf):
            scene = scene[:-len(suf)]
            break
    return ALIAS.get(scene, scene)


@functools.lru_cache(maxsize=None)
def resolve(scene):
    s = _norm(scene)
    for fam in FAMILIES:
        d = os.path.join(C.maps_dir(), fam, s)
        if os.path.isdir(d):
            return fam, d
    raise KeyError(f"no map for scene '{scene}' under {C.maps_dir()}")


def is_metric(scene):
    return resolve(scene)[0] == "eth-ucy"


@functools.lru_cache(maxsize=None)
def load_seg(scene):
    return np.array(Image.open(os.path.join(resolve(scene)[1], "seg.png")).convert("L"))


@functools.lru_cache(maxsize=None)
def load_walkable(scene):
    return np.array(Image.open(os.path.join(resolve(scene)[1], "walkable.png")).convert("L"))


@functools.lru_cache(maxsize=None)
def homography(scene):
    p = os.path.join(resolve(scene)[1], "H.txt")
    return np.loadtxt(p) if os.path.exists(p) else None


def to_map_pixels(coords, scene):
    coords = np.asarray(coords, dtype=float)
    if not is_metric(scene):
        return coords
    c = coords.reshape(-1, 2)
    img = (np.linalg.inv(homography(scene)) @ np.stack([c[:, 0], c[:, 1], np.ones_like(c[:, 0])], -1).T).T
    return (img / img[:, [2]])[:, :2].reshape(coords.shape)


def seg_to_onehot(seg):
    return np.eye(NUM_SEG_CLASSES, dtype=np.float32)[seg].transpose(2, 0, 1)
