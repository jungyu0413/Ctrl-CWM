import numpy as np
import torch
from CrowdES.inference_model import CrowdESFramework, batched_nearest_nonzero_idx_kdtree
from utils.homography import image2world, world2image

from .. import config as C
from ..data import maps as MU
from ..data.grid import grid2meter, meter2grid, scene_grid, soft_argmax
from ..data.prediction import render_ctx


class CtrlCWMSimulator(CrowdESFramework):
    """CrowdES harness (emitter, agent lifecycle, metrics) with Ctrl-CWM as the crowd transition (Eq. 2).

    Every agent keeps a local navigation goal decoded from the goal heatmap and refreshed every L steps.
    At each step the actor proposes a displacement for all agents and the planner selects the executed one.
    Agents leave at the emitter's destination or at the scene boundary."""

    def __init__(self, config, model, planner, control_onset=0.0, walk_pen=12.0):
        super().__init__(config)
        self.wm = model.eval()
        self.planner = planner
        self.control_onset = control_onset
        self.walk_pen = walk_pen
        self.scene_stem = None

    def generate(self, scenario_len, seed=None):
        self.local_goals = {}
        self.control_frame = self.control_onset * scenario_len / self.dataset_fps * self.simulator_fps
        return super().generate(scenario_len, seed)

    def _to_grid(self, world):
        if MU.is_metric(self.scene_stem):
            return meter2grid(world, self.scene_stem)
        _, H0, W0 = scene_grid(self.scene_stem)
        px = np.asarray(world2image(np.asarray(world, np.float32), self.H), np.float32)
        return px * np.array([C.GRID / W0, C.GRID / H0], np.float32)

    def _from_grid(self, g):
        if MU.is_metric(self.scene_stem):
            return grid2meter(g, self.scene_stem)
        _, H0, W0 = scene_grid(self.scene_stem)
        return image2world(np.asarray(g, np.float32) * np.array([W0 / C.GRID, H0 / C.GRID], np.float32), self.H)

    def _goals(self, ids, f, scene):
        """Local goal g_i: decoded at the start of each agent's L-step window and held within it."""
        stale = [i for i, a in enumerate(ids) if a not in self.local_goals or self.local_goals[a][1] >= C.PRED]
        if stale:
            logits = self.wm.goal_logits(f)[stale] + (scene[stale, 0] - 1.0) * self.walk_pen
            for i, g in zip(stale, soft_argmax(logits)):
                self.local_goals[ids[i]] = [g, 0]
        goal = torch.stack([self.local_goals[a][0] for a in ids])
        for a in ids:
            self.local_goals[a][1] += 1
        return goal

    @torch.no_grad()
    def _rollout(self, hist_m, T_fut):
        m = self.wm
        ids = list(self.agent_ids_in_current_scene)
        A = len(ids)
        hist = torch.as_tensor(self._to_grid(np.asarray(hist_m, np.float32)), dtype=torch.float32, device=C.DEV)
        scene = scene_grid(self.scene_stem)[0][None].expand(A, -1, -1, -1)
        sse = torch.tensor([[0, A]], device=C.DEV)
        allow = torch.zeros(A, A, device=C.DEV)
        control = self.current_frame >= self.control_frame
        for _ in range(T_fut):
            focal, crowd, pos, vel_hist = render_ctx(hist, hist.shape[1] - 1, sse)
            f = m.features(scene, focal, crowd, pos, vel_hist, allow)
            goal = self._goals(ids, f, scene)
            cur, vel = hist[:, -1], hist[:, -1] - hist[:, -2]
            base = m.act(f, cur, goal, vel)
            action = self.planner(m, f, cur, base, goal, vel_hist, allow, control)
            hist = torch.cat([hist, (cur + action).clamp(0, C.GRID - 1)[:, None]], 1)
        return np.asarray(self._from_grid(hist[:, -T_fut:].cpu().numpy()), np.float32)

    def process_simulator(self):
        T_hist = self.config.crowd_simulator.simulator.history_length
        T_fut = self.traj_fut_frame
        before = len(self.agent_ids_in_current_scene)
        for recurrent in range(self.window_frame // T_fut):
            for aid in self.new_agent_ids:
                if T_fut * recurrent <= self.agent_parameter[aid]["frame_origin"] < T_fut * (recurrent + 1):
                    self.agent_ids_in_current_scene.append(aid)
            if not self.agent_ids_in_current_scene:
                continue
            B = len(self.agent_ids_in_current_scene)
            traj_hist, dest = np.zeros((B, T_hist, 2)), np.zeros((B, 2))
            for i, aid in enumerate(self.agent_ids_in_current_scene):
                tr = self.agent_trajectory[aid]
                if len(tr) < T_hist:
                    traj_hist[i] = np.array(self.agent_parameter[aid]["origin_xy"])[None]
                    if len(tr):
                        traj_hist[i, -len(tr):] = tr
                else:
                    traj_hist[i] = tr[-T_hist:]
                dest[i] = self.agent_parameter[aid]["goal_xy"]
            fut_m = self._rollout(image2world(traj_hist, self.H), T_fut)
            dest_m = image2world(dest, self.H)
            fut = batched_nearest_nonzero_idx_kdtree(self.kdtree, world2image(fut_m, self.H))

            for i, aid in list(enumerate(self.agent_ids_in_current_scene)):
                if aid in self.new_agent_ids:
                    fo = min(max(0, self.agent_parameter[aid]["frame_origin"] - T_fut * recurrent), T_fut - 1)
                    tr = fut[i, :T_fut - fo]
                    self.new_agent_ids.remove(aid)
                else:
                    tr = np.concatenate([self.agent_trajectory[aid], fut[i]], 0)
                out = ((fut[i] <= 0) | (fut[i] >= np.array(self.scene_size)[[1, 0]] - 1)).any(1)
                dist = np.linalg.norm(fut_m[i] - dest_m[i][None], axis=1)
                if out.any():
                    tr = tr[:-(T_fut - np.where(out)[0][0])]
                    self.agent_ids_in_current_scene.remove(aid)
                elif (dist < 0.5).any():
                    cut = T_fut - np.argmin(dist) + 1
                    tr = tr[:-cut] if cut > 0 else tr
                    self.agent_ids_in_current_scene.remove(aid)
                self.agent_trajectory[aid] = tr
        self.statistics_dropped = self.statistics_added + before - len(self.agent_ids_in_current_scene)
