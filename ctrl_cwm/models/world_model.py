import torch
import torch.nn as nn

from .. import config as C
from .actor import Actor, sample_local
from .critic import Critic
from .encoders import FeatureExtractor, GoalEncoder
from .heads import GoalDecoder, WaypointDecoder


class CtrlCWM(nn.Module):
    """Feature extractor h_theta with its prediction heads (stage 1) and the actor pi_psi (stage 2)."""

    def __init__(self, horizon=C.PRED, width=1.0, use_interaction=True):
        super().__init__()
        self.width = float(width)

        def w(c):
            return max(8, int(round(c * self.width / 8.0)) * 8)

        self.extractor = FeatureExtractor(w, use_interaction)
        self.goal_decoder = GoalDecoder(w)
        self.goal_encoder = GoalEncoder(w)
        self.waypoint_decoder = WaypointDecoder(w, horizon)
        self.actor = Actor(w(32))

    def features(self, scene, focal_hm, crowd_hm, pos, vel_hist, allow):
        return self.extractor(scene, focal_hm, crowd_hm, pos, vel_hist, allow)

    def goal_logits(self, f):
        return self.goal_decoder(f.bottleneck, f.skips)

    def waypoint_logits(self, f, goal_hm):
        """-> (waypoint heatmaps [A, L, G, G], next-position heatmap [A, G, G])."""
        return self.waypoint_decoder(f.bottleneck, f.skips, self.goal_encoder(goal_hm))

    def act(self, f, pos, goal, vel):
        """Actor proposal for every agent: pos, goal, vel [A, 2] -> displacement [A, 2]."""
        return self.actor(sample_local(f.local_map, pos), f.interaction, pos, goal, vel)

    def act_imagined(self, local_map, pos, goal, vel, vel_hist, allow):
        """Actor over W imagined worlds with cached scene maps: pos, vel [W, A, 2] -> [W, A, 2]."""
        c = self.extractor.interaction_embedding(pos / C.GRID, vel_hist, allow)
        return self.actor(sample_local(local_map, pos), c, pos, goal, vel)

    def freeze_world_model(self):
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith("actor.")


def save_checkpoint(path, model, critic=None, **meta):
    torch.save({"model": model.state_dict(), "width": model.width,
                "use_interaction": model.extractor.use_interaction,
                "critic": critic.state_dict() if critic is not None else None,
                "grid": C.GRID, "sig": C.SIG, **meta}, path)


def load_checkpoint(path, device=C.DEV):
    """-> (CtrlCWM, Critic or None, checkpoint dict)."""
    ck = torch.load(path, map_location=device)
    model = CtrlCWM(width=ck["width"], use_interaction=ck.get("use_interaction", True)).to(device)
    model.load_state_dict(ck["model"])
    critic = None
    if ck.get("critic") is not None:
        critic = Critic().to(device)
        critic.load_state_dict(ck["critic"])
    return model, critic, ck
