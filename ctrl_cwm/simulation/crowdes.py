import os
import sys

_installed = False


def setup(root=None, compat=True):
    root = os.path.abspath(root or os.environ.get("CROWDES_ROOT", ""))
    if not os.path.isdir(os.path.join(root, "CrowdES")):
        raise FileNotFoundError(f"CrowdES repository not found at '{root}' (use --crowdes-root or CROWDES_ROOT)")
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    if compat:
        install_compat()
    return root


def install_compat():
    """PyTorch >= 2 guards for emitter windows that propose zero agents (no effect otherwise)."""
    global _installed
    if _installed:
        return
    import torch
    import torch.nn as nn

    _orig_init = nn.TransformerEncoder.__init__

    def _no_nested_init(self, *args, **kwargs):
        kwargs["enable_nested_tensor"] = False
        _orig_init(self, *args, **kwargs)
        self.enable_nested_tensor = False

    nn.TransformerEncoder.__init__ = _no_nested_init

    from CrowdES.emitter.emitter_pipeline import CrowdESEmitterPipeline
    _orig_call = CrowdESEmitterPipeline.__call__

    def _guarded_call(self, *args, **kwargs):
        n = kwargs.get("num_crowd", None)
        if n is not None and int(n) < 1:
            kwargs["num_crowd"] = 1
        out = _orig_call(self, *args, **kwargs)
        try:
            ce = out["crowd_emission"]
            if (~torch.isfinite(ce)).any():
                out["crowd_emission"] = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        return out

    CrowdESEmitterPipeline.__call__ = _guarded_call
    _installed = True


def load_dataset(name, split):
    from utils.config import get_config
    from utils.dataloader.evaluation_dataloader import EvaluationDataset
    cfg = get_config(f"./configs/model/CrowdES_{name}.yaml", None, None)
    return cfg, EvaluationDataset(cfg, split)
