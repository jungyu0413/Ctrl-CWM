import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .. import config as C
from ..data.prediction import allow_mask, render_ctx
from ..models import Critic, Discriminator, critic_features, save_checkpoint
from .losses import collision_count, imitation_loss, kinematic_loss


@dataclass
class BehaviorConfig:
    epochs: int = 40
    windows_per_epoch: int = 256
    batch: int = 4
    horizon: int = C.PRED
    lr: float = 3e-4
    lr_d: float = 2e-4
    w_pos: float = 1.0
    w_dtw: float = 0.5
    dtw_gamma: float = 1.0
    w_adv: float = 0.3
    actor_kin: bool = True
    calib_epochs: int = 8
    calib_ratio: float = 0.3
    critic_steps: int = 2
    candidates: int = 16
    cand_sigma: float = 1.5
    critic_horizon: int = 4
    focal_per_window: int = 2
    teach_kin: bool = True
    teach_coll: bool = True
    coll_radius: float = 4.0
    tau: float = 1.0
    critic_loss: str = "listwise"
    select_tol: float = 0.05


def spearman(q, y):
    rq = q.argsort(-1).argsort(-1).float()
    ry = y.argsort(-1).argsort(-1).float()
    rq, ry = rq - rq.mean(-1, keepdim=True), ry - ry.mean(-1, keepdim=True)
    return ((rq * ry).sum(-1) / (rq.norm(dim=-1) * ry.norm(dim=-1) + 1e-9)).mean().item()


class BehaviorLearner:
    """Stage 2: actor (Eq. 4-6) and critic (Eq. 7-9) on the frozen world model."""

    def __init__(self, model, scenes, cfg: BehaviorConfig):
        from ..data import simulation as SD
        self.SD, self.m, self.scenes, self.cfg = SD, model, scenes, cfg
        model.freeze_world_model()
        self.D = Discriminator().to(C.DEV)
        self.critic = Critic().to(C.DEV)
        self.opt_actor = torch.optim.AdamW(model.actor.parameters(), cfg.lr, weight_decay=1e-5)
        self.opt_d = torch.optim.Adam(self.D.parameters(), cfg.lr_d, betas=(0.5, 0.9))
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), cfg.lr, weight_decay=1e-5)
        self.w_kin = self.w_tkin = self.w_coll = None
        self.calib = {k: [] for k in ("g_base", "g_kin", "s_imit", "s_kin", "s_coll")}

    def rollout(self, scene, hist, fut, mask, sse, first=None):
        """All agents follow the actor for L steps from hist [A, T, 2]; first [A, 2] replaces the first action.
        Features are recomputed from the rolled-out state at every step; gradients flow through positions."""
        m, SD = self.m, self.SD
        allow = allow_mask(sse.tolist(), hist.shape[0])
        li = SD.last_alive(mask)
        h, pos, vel = hist, hist[:, -1], hist[:, -1] - hist[:, -2]
        preds = []
        for k in range(self.cfg.horizon):
            with torch.no_grad():
                focal, crowd, p, v = render_ctx(h.detach(), h.shape[1] - 1, sse)
                f = m.features(scene, focal, crowd, p, v, allow)
            a = first if k == 0 and first is not None else m.act(f, pos, SD.waypoint_goal(k, fut, li), vel)
            pos, vel = pos + a, a
            preds.append(pos)
            h = torch.cat([h, pos[:, None]], 1)
        return torch.stack(preds, 1)

    def actor_step(self, epoch, scene, hist, fut, mask, sse):
        cfg = self.cfg
        traj = self.rollout(scene, hist, fut, mask, sse)
        gt = fut[:, :cfg.horizon]
        valid = mask[:, :cfg.horizon].prod(1)
        if valid.sum() < 2:
            return None
        w = valid / valid.sum()
        disp = torch.diff(torch.cat([hist[:, -1:], traj], 1), dim=1) / C.GRID
        disp_gt = torch.diff(torch.cat([hist[:, -1:], gt], 1), dim=1) / C.GRID

        self.opt_d.zero_grad()
        (((self.D(disp_gt) - 1) ** 2 * w).sum() + (self.D(disp.detach()) ** 2 * w).sum()).backward()
        self.opt_d.step()

        imit = (imitation_loss(traj, gt, cfg.w_pos, cfg.w_dtw, cfg.dtw_gamma) * w).sum()
        adv = (((self.D(disp) - 1) ** 2) * w).sum()
        kin = (kinematic_loss(traj, gt) * w).sum()
        loss = imit + cfg.w_adv * adv
        if cfg.actor_kin and epoch < cfg.calib_epochs:
            params = list(self.m.actor.parameters())
            for key, term in (("g_base", loss), ("g_kin", kin)):
                g = torch.autograd.grad(term, params, retain_graph=True)
                self.calib[key].append(torch.sqrt(sum((x ** 2).sum() for x in g)).item())
        elif cfg.actor_kin:
            loss = loss + self.w_kin * kin
        self.opt_actor.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.m.actor.parameters(), 1.0)
        self.opt_actor.step()
        return imit.item(), adv.item()

    @torch.no_grad()
    def candidate_costs(self, scene, hist, fut, mask, focal):
        """For each focal agent, K candidates replace its first action; the actor supplies every other action.
        -> costs [F, K] over L steps (Eq. 7) and critic features [F, K, H, 6] over the first H steps."""
        cfg, m, SD = self.cfg, self.m, self.SD
        n, Fn, K, L = hist.shape[0], len(focal), cfg.candidates, cfg.horizon
        sse = torch.tensor([[0, n]], device=C.DEV)
        fh, crowd, p, v = render_ctx(hist, hist.shape[1] - 1, sse)
        f = m.features(scene, fh, crowd, p, v, torch.zeros(n, n, device=C.DEV))
        li = SD.last_alive(mask)
        goal = SD.waypoint_goal(0, fut, li)
        base = m.act(f, hist[:, -1], goal, hist[:, -1] - hist[:, -2])
        first = base[None, None].repeat(Fn, K, 1, 1)
        first[torch.arange(Fn)[:, None], torch.arange(K)[None], focal[:, None]] = \
            base[focal][:, None] + cfg.cand_sigma * torch.randn(Fn, K, 2, device=C.DEV)

        W = Fn * K
        def rep(x):
            return x[None].expand(W, *x.shape).reshape(W * n, *x.shape[1:])
        sse_w = torch.tensor([[i * n, (i + 1) * n] for i in range(W)], device=C.DEV)
        traj = self.rollout(rep(scene), rep(hist), rep(fut), rep(mask), sse_w, first.reshape(W * n, 2))
        traj = traj.reshape(Fn, K, n, L, 2)

        tf = traj[torch.arange(Fn), :, focal].reshape(W, L, 2)
        gt = fut[focal, :L][:, None].expand(Fn, K, L, 2).reshape(W, L, 2)
        imit = imitation_loss(tf, gt, cfg.w_pos, cfg.w_dtw, cfg.dtw_gamma).reshape(Fn, K)
        kin = kinematic_loss(tf, gt).reshape(Fn, K)
        coll = torch.stack([collision_count(traj[i], int(focal[i]), cfg.coll_radius) for i in range(Fn)])
        for key, x in (("s_imit", imit), ("s_kin", kin), ("s_coll", coll)):
            self.calib[key].append(x.std(1).mean().item())
        cost = imit.clone()
        if cfg.teach_kin and self.w_tkin is not None:
            cost = cost + self.w_tkin * kin
        if cfg.teach_coll and self.w_coll is not None:
            cost = cost + self.w_coll * coll

        H = cfg.critic_horizon
        path = torch.cat([hist[:, -1][None, None, :, None].expand(Fn, K, n, 1, 2), traj[..., :H, :]], 3)
        feats = critic_features(traj[..., :H, :].reshape(W, n, H, 2),
                                torch.diff(path, dim=3).reshape(W, n, H, 2), goal)
        feats = feats.reshape(Fn, K, n, H, -1)[torch.arange(Fn), :, focal]
        return cost, feats

    def critic_step(self, scene, hist, fut, mask, sse):
        cfg = self.cfg
        valid = mask[:, :cfg.horizon].prod(1)
        losses, rhos = [], []
        for s, e in sse.tolist():
            alive = (valid[s:e] > 0).nonzero().flatten()
            if e - s < 2 or len(alive) == 0:
                continue
            focal = alive[torch.randperm(len(alive), device=alive.device)[:cfg.focal_per_window]]
            cost, feats = self.candidate_costs(scene[s:e], hist[s:e], fut[s:e], mask[s:e], focal)
            q = self.critic(feats.reshape(-1, *feats.shape[2:])).reshape(cost.shape)
            if cfg.critic_loss == "listwise":
                losses.append(-(F.softmax(-cost / cfg.tau, 1) * F.log_softmax(q / cfg.tau, 1)).sum(1).mean())
            else:
                losses.append(F.mse_loss(q, -cost))
            rhos.append(spearman(q.detach(), -cost))
        if not losses:
            return None
        loss = torch.stack(losses).mean()
        self.opt_critic.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()
        return loss.item(), float(np.mean(rhos))

    def calibrate(self):
        c, r = self.calib, self.cfg.calib_ratio
        if self.cfg.actor_kin and c["g_kin"]:
            self.w_kin = r * np.mean(c["g_base"]) / max(np.mean(c["g_kin"]), 1e-9)
        if c["s_imit"]:
            self.w_tkin = r * np.mean(c["s_imit"]) / max(np.mean(c["s_kin"]), 1e-9)
            self.w_coll = r * np.mean(c["s_imit"]) / max(np.mean(c["s_coll"]), 1e-9)
        return self.w_kin, self.w_tkin, self.w_coll

    @torch.no_grad()
    def evaluate(self, rng, n=48):
        SD, cfg = self.SD, self.cfg
        self.m.eval()
        wins = SD.make_windows(self.scenes, rng, n, cfg.horizon)
        err = cnt = 0.
        for i in range(0, len(wins), cfg.batch):
            scene, hist, fut, mask, sse = SD.stack_batch(self.scenes, wins[i:i + cfg.batch])
            traj = self.rollout(scene, hist, fut, mask, sse)
            err += ((traj - fut).norm(dim=-1) * mask).sum().item()
            cnt += mask.sum().item()
        return err / max(cnt, 1)

    def fit(self, rng, ckpt_path, log=print):
        cfg, SD = self.cfg, self.SD
        best_ade, sel = float("inf"), None
        for epoch in range(cfg.epochs):
            t0 = time.time()
            self.m.actor.train(); self.D.train(); self.critic.train()
            wins = SD.make_windows(self.scenes, rng, cfg.windows_per_epoch, cfg.horizon)
            stats = {"imit": [], "adv": [], "critic": [], "rho": []}
            for i in range(0, len(wins), cfg.batch):
                batch = SD.stack_batch(self.scenes, wins[i:i + cfg.batch])
                for _ in range(cfg.critic_steps):
                    out = self.critic_step(*batch)
                    if out:
                        stats["critic"].append(out[0]); stats["rho"].append(out[1])
                out = self.actor_step(epoch, *batch)
                if out:
                    stats["imit"].append(out[0]); stats["adv"].append(out[1])
            if epoch + 1 == cfg.calib_epochs:
                log("calibrated weights actor_kin={} teacher_kin={} teacher_coll={}".format(*self.calibrate()))

            ade = self.evaluate(np.random.default_rng(123))
            rho = float(np.mean(stats["rho"])) if stats["rho"] else 0.0
            best_ade = min(best_ade, ade)
            qualifies = ade <= best_ade * (1 + cfg.select_tol)
            if qualifies and (sel is None or sel["ade"] > best_ade * (1 + cfg.select_tol) or rho > sel["rho"]):
                sel = {"epoch": epoch + 1, "ade": ade, "rho": rho}
                save_checkpoint(ckpt_path, self.m, self.critic, critic_horizon=cfg.critic_horizon, **sel)
            log(f"epoch {epoch + 1}/{cfg.epochs} {time.time() - t0:.0f}s "
                + " ".join(f"{k}={np.mean(v):.4f}" for k, v in stats.items() if v)
                + f" ade={ade:.3f}" + (" *" if sel and sel["epoch"] == epoch + 1 else ""))
        return sel
