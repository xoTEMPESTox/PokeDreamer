"""Hybrid diffusion world model: symbolic RAM dynamics + RAM-conditioned frame renderer.

Hypothesis: Pixel-only diffusion world models drift in map identity because a 4-frame
pixel context cannot remember location once the visual scene is ambiguous. A hybrid model
couples (a) RamDynamics: a compact symbolic model predicting the next RAM state from
recent RAM states + button actions, and (b) HybridDenoiser: EDM diffusion renderer
conditioned on both pixel context, actions, and the target frame's RAM state.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.dwm.model import (
    AUG_BUCKETS,
    AUG_MAX,
    CTX,
    N_ACTIONS,
    P_MEAN,
    P_STD,
    RHO,
    SIGMA_DATA,
    SIGMA_MAX,
    SIGMA_MIN,
    Denoiser,
    to_float,
)


def _clean_ram(ram, device=None, required=None):
    """Normalize RAM dict keys and datatypes to torch.long on target device."""
    alias = {
        "map": ("map", "map_ids"),
        "x": ("x", "xs"),
        "y": ("y", "ys"),
        "facing": ("facing", "facings"),
        "battle": ("battle", "in_battles"),
    }
    out = {}
    for key, candidates in alias.items():
        val = None
        for c in candidates:
            if c in ram:
                val = ram[c]
                break
        if val is None:
            if required and key in required:
                raise KeyError(f"Missing RAM field '{key}' (or aliases {candidates}) in ram dict.")
            continue
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        elif not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if device is not None:
            val = val.to(device)
        if key == "facing":
            val = val.long()
            if (val > 3).any():
                val = val // 4
        elif key == "battle":
            val = val.long()
        else:
            val = val.long()
        out[key] = val
    return out


class RamDynamics(nn.Module):
    """Symbolic RAM transition model predicting next game state from CTX RAM states and actions."""

    def __init__(self, embed_dim=64, hidden_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.emb_map = nn.Embedding(256, embed_dim)
        self.emb_x = nn.Embedding(64, embed_dim)
        self.emb_y = nn.Embedding(80, embed_dim)
        self.emb_facing = nn.Embedding(4, embed_dim)
        self.emb_battle = nn.Embedding(2, embed_dim)
        self.emb_act = nn.Embedding(N_ACTIONS, embed_dim)

        in_dim = CTX * 6 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head_map = nn.Linear(hidden_dim, 256)
        self.head_x = nn.Linear(hidden_dim, 64)
        self.head_y = nn.Linear(hidden_dim, 80)
        self.head_facing = nn.Linear(hidden_dim, 4)
        self.head_battle = nn.Linear(hidden_dim, 2)

    def forward(self, ram_ctx, actions):
        """ram_ctx: dict of (B, CTX) long tensors; actions: (B, CTX) long tensor."""
        ram = _clean_ram(ram_ctx, device=self.emb_map.weight.device, required=("map", "x", "y", "facing", "battle"))
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.to(device=self.emb_map.weight.device).long()
        embs = [
            self.emb_map(ram["map"]),
            self.emb_x(ram["x"]),
            self.emb_y(ram["y"]),
            self.emb_facing(ram["facing"]),
            self.emb_battle(ram["battle"]),
            self.emb_act(actions),
        ]
        feat = torch.cat(embs, dim=-1).flatten(1)
        h = self.mlp(feat)
        return {
            "map": self.head_map(h),
            "x": self.head_x(h),
            "y": self.head_y(h),
            "facing": self.head_facing(h),
            "battle": self.head_battle(h),
        }

    def loss(self, ram_ctx, actions, ram_target=None):
        """Sum of cross-entropy losses for next RAM prediction."""
        if ram_target is None:
            clean = _clean_ram(ram_ctx)
            ctx = {k: v[:, :CTX] for k, v in clean.items()}
            target = {k: v[:, CTX] for k, v in clean.items()}
            act_ctx = actions[:, :CTX]
        else:
            ctx = ram_ctx
            target = _clean_ram(ram_target, device=self.emb_map.weight.device)
            act_ctx = actions[:, :CTX] if actions.shape[1] > CTX else actions

        logits = self.forward(ctx, act_ctx)
        t_clean = _clean_ram(target, device=logits["map"].device)
        return (
            F.cross_entropy(logits["map"], t_clean["map"])
            + F.cross_entropy(logits["x"], t_clean["x"])
            + F.cross_entropy(logits["y"], t_clean["y"])
            + F.cross_entropy(logits["facing"], t_clean["facing"])
            + F.cross_entropy(logits["battle"], t_clean["battle"])
        )

    @torch.no_grad()
    def predict(self, ram_ctx, actions, sample=False):
        """Predict next RAM state from context and actions. Returns dict of (B,) long tensors."""
        logits = self.forward(ram_ctx, actions)
        preds = {}
        for k in ("map", "x", "y", "facing", "battle"):
            lg = logits[k]
            if sample:
                probs = F.softmax(lg, dim=-1)
                preds[k] = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                preds[k] = lg.argmax(dim=-1)
        preds["map_ids"] = preds["map"]
        preds["xs"] = preds["x"]
        preds["ys"] = preds["y"]
        preds["facings"] = preds["facing"]
        preds["in_battles"] = preds["battle"]
        return preds


class HybridDenoiser(Denoiser):
    """Diffusion world model conditioned on pixel context, actions, and predicted target RAM."""

    def __init__(self, cond_dim=256, ram_dim=256, ram_model=None):
        super().__init__(cond_dim=cond_dim)
        self.ram_model = ram_model
        self.ram_map_emb = nn.Embedding(256, ram_dim)
        self.ram_x_emb = nn.Embedding(64, ram_dim)
        self.ram_y_emb = nn.Embedding(80, ram_dim)
        self.ram_battle_emb = nn.Embedding(2, ram_dim)
        self.ram_proj = nn.Linear(ram_dim, cond_dim)

    def ram_emb(self, ram_target):
        ram = _clean_ram(ram_target, device=self.ram_proj.weight.device, required=("map", "x", "y", "battle"))
        m = ram["map"].squeeze(1) if ram["map"].ndim == 2 else ram["map"]
        x = ram["x"].squeeze(1) if ram["x"].ndim == 2 else ram["x"]
        y = ram["y"].squeeze(1) if ram["y"].ndim == 2 else ram["y"]
        b = ram["battle"].squeeze(1) if ram["battle"].ndim == 2 else ram["battle"]
        s = self.ram_map_emb(m) + self.ram_x_emb(x) + self.ram_y_emb(y) + self.ram_battle_emb(b)
        return self.ram_proj(s)

    def raw(self, x_noisy, sigma, ctx, actions, ram_target, aug_bucket=0):
        s = sigma[:, None, None, None]
        c_in = 1 / (s ** 2 + SIGMA_DATA ** 2).sqrt()
        c_skip = SIGMA_DATA ** 2 / (s ** 2 + SIGMA_DATA ** 2)
        c_out = s * SIGMA_DATA / (s ** 2 + SIGMA_DATA ** 2).sqrt()
        b = x_noisy.shape[0]
        if isinstance(aug_bucket, int):
            aug_bucket = torch.full((b,), aug_bucket, device=x_noisy.device, dtype=torch.long)
        cond = (
            self.noise_emb(sigma.log() / 4)
            + self.act_emb(actions).flatten(1)
            + self.ram_emb(ram_target)
            + self.aug_emb(aug_bucket)
        )
        out = self.unet(torch.cat([x_noisy * c_in, ctx], 1), self.cond_mlp(cond))
        return out, c_skip, c_out

    def loss(self, frames, actions, ram_target, ctx_aug=False):
        """Teacher-forced renderer loss on target frame conditioned on true next RAM."""
        b = frames.shape[0]
        target, ctx = frames[:, -1:], frames[:, :-1]
        aug = torch.randint(0, AUG_BUCKETS if ctx_aug else 1, (b,), device=frames.device)
        if ctx_aug:
            ctx = ctx + torch.randn_like(ctx) * (aug * AUG_MAX / (AUG_BUCKETS - 1))[:, None, None, None]
        sigma = (torch.randn(b, device=frames.device) * P_STD + P_MEAN).exp()
        x_noisy = target + torch.randn_like(target) * sigma[:, None, None, None]
        out, c_skip, c_out = self.raw(x_noisy, sigma, ctx, actions, ram_target, aug)
        return F.mse_loss(out.float(), ((target - c_skip * x_noisy) / c_out).float())

    @torch.no_grad()
    def sample(self, ctx, actions, ram_target, steps=3, aug_bucket=0):
        """Sample one frame conditioned on context frames, actions, and predicted target RAM."""
        b = ctx.shape[0]
        aug = torch.full((b,), aug_bucket, device=ctx.device, dtype=torch.long)
        if aug_bucket:
            ctx = ctx + torch.randn_like(ctx) * aug_bucket * AUG_MAX / (AUG_BUCKETS - 1)
        i = torch.arange(steps, device=ctx.device) / max(steps - 1, 1)
        sigmas = (SIGMA_MAX ** (1 / RHO) + i * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        x = torch.randn(b, 1, *ctx.shape[2:], device=ctx.device) * sigmas[0]
        for s, s_next in zip(sigmas[:-1], sigmas[1:]):
            out, c_skip, c_out = self.raw(x, s.expand(b), ctx, actions, ram_target, aug)
            denoised = c_skip * x + c_out * out.float()
            x = x + (x - denoised) / s * (s_next - s)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def rollout(self, frames_ctx, ram_ctx, actions, steps=3, ram_model=None, aug_bucket=0):
        """Open-loop hybrid imagination.
        frames_ctx: (B, CTX, H, W)
        ram_ctx: dict of (B, CTX)
        actions: (B, CTX + K - 1)
        Returns: imagined frames (B, K, H, W) in [-1, 1] and predicted RAM dict of (B, K) int arrays.
        """
        ram_dyn = ram_model if ram_model is not None else self.ram_model
        if ram_dyn is None:
            raise ValueError("ram_model must be provided to rollout either as an argument or attached to HybridDenoiser")

        device = self.ram_proj.weight.device
        if isinstance(frames_ctx, np.ndarray):
            frames_ctx = torch.from_numpy(frames_ctx)
        frames_ctx = frames_ctx.to(device)
        if frames_ctx.dtype == torch.uint8:
            frames_ctx = to_float(frames_ctx)

        clean_ram = _clean_ram(ram_ctx, device=device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.to(device).long()

        k_total = actions.shape[1] - CTX + 1
        curr_frames = frames_ctx
        curr_ram = {k: v.clone() for k, v in clean_ram.items()}

        out_frames = []
        out_ram = {k: [] for k in ("map", "x", "y", "facing", "battle")}

        for k in range(k_total):
            act_slice = actions[:, k : k + CTX]
            next_ram = ram_dyn.predict(curr_ram, act_slice)
            for key in out_ram:
                out_ram[key].append(next_ram[key])

            nxt_frame = self.sample(curr_frames[:, -CTX:], act_slice, next_ram, steps=steps, aug_bucket=aug_bucket)
            out_frames.append(nxt_frame)

            curr_frames = torch.cat([curr_frames[:, 1:], nxt_frame], dim=1)
            for key in ("map", "x", "y", "facing", "battle"):
                curr_ram[key] = torch.cat([curr_ram[key][:, 1:], next_ram[key].unsqueeze(1)], dim=1)

        imagined_frames = torch.cat(out_frames, dim=1)
        pred_ram = {k: torch.stack(v, dim=1).cpu().numpy().astype(np.int64) for k, v in out_ram.items()}
        pred_ram["map_ids"] = pred_ram["map"]
        pred_ram["xs"] = pred_ram["x"]
        pred_ram["ys"] = pred_ram["y"]
        pred_ram["facings"] = pred_ram["facing"]
        pred_ram["in_battles"] = pred_ram["battle"]

        return imagined_frames, pred_ram


def rollout(*args, **kwargs):
    """Convenience wrapper for hybrid rollout."""
    if len(args) > 0 and isinstance(args[0], HybridDenoiser):
        model = args[0]
        if len(args) > 1 and isinstance(args[1], (RamDynamics, nn.Module)):
            ram_model = args[1]
            return model.rollout(*args[2:], ram_model=ram_model, **kwargs)
        return model.rollout(*args[1:], **kwargs)

    frames_ctx = args[0]
    ram_ctx = args[1]
    actions = args[2]
    model = kwargs.pop("model", None)
    if model is None:
        raise ValueError("model (HybridDenoiser) must be provided to rollout")
    return model.rollout(frames_ctx, ram_ctx, actions, **kwargs)
