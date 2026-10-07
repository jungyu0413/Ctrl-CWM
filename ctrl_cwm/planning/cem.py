from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .. import config as C
from ..models import critic_features


@dataclass
class UserCost:
    """c_user: fraction of imagined agent-steps inside avoid discs (Eq. 11), or mean distance to a target (Eq. 12)."""
    avoid: List[Tuple[float, float, float]] = field(default_factory=list)
    attract: Optional[Tuple[float, float]] = None

    def __call__(self, pos):
        """pos [W, A, H, 2] -> [W]."""
        cost = torch.zeros(pos.shape[0], device=pos.device)
        if self.avoid:
            inside = torch.zeros(pos.shape[:-1], dtype=torch.bool, device=pos.device)
            for cx, cy, r in self.avoid:
                inside |= (pos - pos.new_tensor([cx, cy])).norm(dim=-1) <= r
            cost = cost + inside.float().mean((1, 2))
        if self.attract is not None:
            cost = cost + ((pos - pos.new_tensor(self.attract)).norm(dim=-1) / C.GRID).mean((1, 2))
        return cost


@dataclass
class PlannerConfig:
    mode: str = "critic"
    candidates: int = 16
    iters: int = 4
    horizon: int = 4
    sigma: float = 1.5
    elite: float = 0.25
    w_user: float = 0.0


class CEMPlanner:
    """Algorithm 1. Each agent's next displacement is optimized in parallel: candidate k replaces only that agent's
    first action, the actor supplies every other action over an H-step imagined rollout, and the candidate is scored
    by J = V(imagined states, i) - w_user * c_user (Eq. 10).
    mode: critic (full model) | user (user cost only) | none (actor only)."""

    def __init__(self, cfg: PlannerConfig, critic=None, user_cost: Optional[UserCost] = None):
        self.cfg, self.critic, self.user_cost = cfg, critic, user_cost

    @torch.no_grad()
    def __call__(self, model, f, cur, base, goal, vel_hist, allow, control=True):
        """cur, base, goal [A, 2] grid px; f features of the current state; returns displacements [A, 2]."""
        cfg = self.cfg
        use_user = control and self.user_cost is not None and cfg.w_user > 0
        if cfg.mode == "none" or (cfg.mode == "user" and not use_user):
            return base
        A, K, H = cur.shape[0], cfg.candidates, cfg.horizon
        n_elite = max(1, int(cfg.elite * K))
        agents = torch.arange(A, device=cur.device)
        mu, std = base.clone(), torch.full_like(base, cfg.sigma)
        for _ in range(cfg.iters):
            cand = mu[:, None] + std[:, None] * torch.randn(A, K, 2, device=cur.device)
            first = base[None, None].repeat(A, K, 1, 1)
            first[agents[:, None], torch.arange(K, device=cur.device)[None], agents[:, None]] = cand
            pos, vel = self.imagine(model, f, cur, first.reshape(A * K, A, 2), goal, vel_hist, allow)
            score = torch.zeros(A, K, device=cur.device)
            if cfg.mode == "critic":
                feats = critic_features(pos, vel, goal).reshape(A, K, A, H, -1)[agents, :, agents]
                score = score + self.critic(feats.reshape(A * K, H, -1)).reshape(A, K)
            if use_user:
                score = score - cfg.w_user * self.user_cost(pos).reshape(A, K)
            elite = cand[agents[:, None], score.topk(n_elite, dim=1).indices]
            mu, std = elite.mean(1), elite.std(1) + 1e-2
        return mu

    def imagine(self, model, f, cur, first, goal, vel_hist, allow):
        """H-step rollout of W worlds [W, A, 2] with scene maps cached at the current state; interaction
        embeddings and local sampling follow the imagined positions."""
        pos = (cur[None] + first).clamp(0, C.GRID - 1)
        vel = first
        ps, vs = [pos], [vel]
        for _ in range(self.cfg.horizon - 1):
            vel = model.act_imagined(f.local_map, pos, goal, vel, vel_hist, allow)
            pos = (pos + vel).clamp(0, C.GRID - 1)
            ps.append(pos); vs.append(vel)
        return torch.stack(ps, 2), torch.stack(vs, 2)
