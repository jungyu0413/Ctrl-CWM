import argparse
import random

import numpy as np
import torch
import yaml


def parse(parser, argv=None):
    """argparse with an optional --config YAML whose keys replace the defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    parser.add_argument("--config", default=None)
    if known.config:
        with open(known.config) as f:
            values = yaml.safe_load(f) or {}
        unknown = sorted(set(values) - {a.dest for a in parser._actions})
        if unknown:
            raise SystemExit(f"{known.config}: unknown keys {unknown}")
        for action in parser._actions:
            if action.dest in values:
                action.required = False
        parser.set_defaults(**values)
    return parser.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def floats(text):
    return tuple(float(x) for x in str(text).split(","))
