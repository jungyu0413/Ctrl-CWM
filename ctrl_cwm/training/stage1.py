import torch
import torch.nn.functional as F

from .. import config as C
from ..data.grid import heatmaps, soft_argmax
from ..data.prediction import build, grid_to_native, sample_goals
from .losses import heatmap_loss

POS_WEIGHT = 50.0


def prediction_loss(model, inputs, w_way, w_dyn):
    """Eq. 3: goal + lambda_way * waypoints + lambda_dyn * next position, GT goal as conditioning."""
    scene, focal, crowd, gtj, pos, vel, allow = inputs[:7]
    pw = torch.tensor(POS_WEIGHT, device=C.DEV)
    f = model.features(scene, focal, crowd, pos, vel, allow)
    goal, way, nxt = gtj[:, C.T - 1], gtj[:, C.OBS:C.T], gtj[:, C.OBS]
    way_logits, dyn_logits = model.waypoint_logits(f, heatmaps(goal, C.SIG))
    l_goal = heatmap_loss(model.goal_logits(f), goal, pw)
    l_way = heatmap_loss(way_logits, way, pw)
    l_dyn = F.binary_cross_entropy_with_logits(dyn_logits, heatmaps(nxt, C.SIG), pos_weight=pw)
    return l_goal + w_way * l_way + w_dyn * l_dyn, (l_goal.item(), l_way.item(), l_dyn.item())


def train_epoch(model, loader, opt, accum=1, w_way=1.0, w_dyn=0.5):
    model.train()
    total, n = torch.zeros(3), 0
    opt.zero_grad()
    for i, b in enumerate(loader):
        loss, parts = prediction_loss(model, build(b, train=True), w_way, w_dyn)
        (loss / accum).backward()
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        total += torch.tensor(parts)
        n += 1
    return (total / max(n, 1)).tolist()


@torch.no_grad()
def evaluate(model, loader, k=20):
    """Best-of-k ADE / FDE with goals sampled from the goal decoder."""
    model.eval()
    s_ade = s_fde = n = 0.
    for b in loader:
        scene, focal, crowd, _, pos, vel, allow, traj, sse, scenes = build(b)
        f = model.features(scene, focal, crowd, pos, vel, allow)
        goals = sample_goals(model.goal_logits(f), k)
        gt = traj[:, C.OBS:C.T]
        ade = torch.full((gt.shape[0],), 1e9, device=C.DEV)
        fde = torch.full_like(ade, 1e9)
        for j in range(k):
            way, _ = model.waypoint_logits(f, heatmaps(goals[:, j], C.SIG))
            err = (grid_to_native(soft_argmax(way), sse, scenes) - gt).norm(dim=-1)
            ade = torch.minimum(ade, err.mean(-1))
            fde = torch.minimum(fde, err[:, -1])
        s_ade += ade.sum().item(); s_fde += fde.sum().item(); n += gt.shape[0]
    return s_ade / n, s_fde / n
