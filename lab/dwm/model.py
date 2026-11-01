"""Diffusion world model (DIAMOND-style EDM denoiser) for 72x80 greyscale Pokemon Red frames.

Next frame f_{t+1} is denoised conditioned on the last CTX frames f_{t-CTX+1..t} (channel-concat) and
actions a_{t-CTX+1..t} (a_t is the button pressed after f_t, so it produces f_{t+1}).
EDM preconditioning and sampler: Karras et al. 2022; settings follow DIAMOND (Alonso et al. 2024).
Context-noise augmentation (noise added to the context frames, level given to the net) follows GameNGen
(Valevski et al. 2024); it trains the model to recover from its own imperfect frames during rollouts.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CTX = 4
N_ACTIONS = 8
SIGMA_DATA = 0.5
P_MEAN, P_STD = -0.4, 1.2                   # training noise distribution: ln sigma ~ N(P_MEAN, P_STD)
SIGMA_MIN, SIGMA_MAX, RHO = 2e-3, 5.0, 7.0  # sampler schedule
AUG_BUCKETS, AUG_MAX = 10, 0.7              # context noise levels: bucket b -> sigma b * AUG_MAX / 9


def to_float(frames_u8):
    return frames_u8.float() / 127.5 - 1.0


class FourierFeatures(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("w", torch.randn(1, dim // 2))

    def forward(self, x):
        f = 2 * math.pi * x[:, None] * self.w
        return torch.cat([f.cos(), f.sin()], 1)


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).transpose(-1, -2).unbind(1)
        y = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(y)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cond_dim, attn=False):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, cout)
        self.cond = nn.Linear(cond_dim, 2 * cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.attn = Attention(cout) if attn else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(cond)[:, :, None, None].chunk(2, 1)
        h = self.conv2(F.silu(self.norm2(h) * (1 + scale) + shift))
        return self.attn(self.skip(x) + h)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, chans=(32, 64, 128, 256), blocks=2, cond_dim=256):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, chans[0], 3, padding=1)
        self.down, skips, c = nn.ModuleList(), [chans[0]], chans[0]
        last = len(chans) - 1
        for i, ch in enumerate(chans):
            for _ in range(blocks):
                self.down.append(ResBlock(c, ch, cond_dim, attn=i == last))
                c = ch
                skips.append(c)
            if i < last:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                skips.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, cond_dim, attn=True), ResBlock(c, c, cond_dim)])
        self.up = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            for _ in range(blocks + 1):
                self.up.append(ResBlock(c + skips.pop(), ch, cond_dim, attn=i == last))
                c = ch
            if i > 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)))
        self.norm_out = nn.GroupNorm(32, c)
        self.conv_out = nn.Conv2d(c, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, cond):
        h = self.conv_in(x)
        hs = [h]
        for m in self.down:
            h = m(h, cond) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        for m in self.mid:
            h = m(h, cond)
        for m in self.up:
            h = m(torch.cat([h, hs.pop()], 1), cond) if isinstance(m, ResBlock) else m(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class Denoiser(nn.Module):
    def __init__(self, cond_dim=256):
        super().__init__()
        self.noise_emb = FourierFeatures(cond_dim)
        self.act_emb = nn.Embedding(N_ACTIONS, cond_dim // CTX)
        self.aug_emb = nn.Embedding(AUG_BUCKETS, cond_dim)
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.unet = UNet(1 + CTX, 1, cond_dim=cond_dim)

    def raw(self, x_noisy, sigma, ctx, actions, aug_bucket):
        """Network output F and EDM scalings; x_noisy (B,1,H,W), sigma (B,), ctx (B,CTX,H,W), actions (B,CTX)."""
        s = sigma[:, None, None, None]
        c_in = 1 / (s ** 2 + SIGMA_DATA ** 2).sqrt()
        c_skip = SIGMA_DATA ** 2 / (s ** 2 + SIGMA_DATA ** 2)
        c_out = s * SIGMA_DATA / (s ** 2 + SIGMA_DATA ** 2).sqrt()
        cond = self.noise_emb(sigma.log() / 4) + self.act_emb(actions).flatten(1) + self.aug_emb(aug_bucket)
        out = self.unet(torch.cat([x_noisy * c_in, ctx], 1), self.cond_mlp(cond))
        return out, c_skip, c_out

    def loss(self, frames, actions):
        """frames (B,CTX+1,H,W) float in [-1,1] (last one is the target), actions (B,CTX) long."""
        b = frames.shape[0]
        target, ctx = frames[:, -1:], frames[:, :-1]
        aug = torch.randint(0, AUG_BUCKETS, (b,), device=frames.device)
        ctx = ctx + torch.randn_like(ctx) * (aug * AUG_MAX / (AUG_BUCKETS - 1))[:, None, None, None]
        sigma = (torch.randn(b, device=frames.device) * P_STD + P_MEAN).exp()
        x_noisy = target + torch.randn_like(target) * sigma[:, None, None, None]
        out, c_skip, c_out = self.raw(x_noisy, sigma, ctx, actions, aug)
        return F.mse_loss(out.float(), ((target - c_skip * x_noisy) / c_out).float())

    @torch.no_grad()
    def sample(self, ctx, actions, steps=3, aug_bucket=0):
        """One next frame (B,1,H,W) in [-1,1] by deterministic Euler sampling (DIAMOND default: 3 steps)."""
        b = ctx.shape[0]
        aug = torch.full((b,), aug_bucket, device=ctx.device, dtype=torch.long)
        if aug_bucket:
            ctx = ctx + torch.randn_like(ctx) * aug_bucket * AUG_MAX / (AUG_BUCKETS - 1)
        i = torch.arange(steps, device=ctx.device) / max(steps - 1, 1)
        sigmas = (SIGMA_MAX ** (1 / RHO) + i * (SIGMA_MIN ** (1 / RHO) - SIGMA_MAX ** (1 / RHO))) ** RHO
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        x = torch.randn(b, 1, *ctx.shape[2:], device=ctx.device) * sigmas[0]
        for s, s_next in zip(sigmas[:-1], sigmas[1:]):
            out, c_skip, c_out = self.raw(x, s.expand(b), ctx, actions, aug)
            denoised = c_skip * x + c_out * out.float()
            x = x + (x - denoised) / s * (s_next - s)
        return x.clamp(-1, 1)

    @torch.no_grad()
    def rollout(self, ctx, actions, steps=3, aug_bucket=0):
        """Open-loop imagination. ctx (B,CTX,H,W) real frames, actions (B,CTX+K-1) -> (B,K,H,W) imagined frames."""
        k_total = actions.shape[1] - CTX + 1
        frames, out = ctx, []
        for k in range(k_total):
            nxt = self.sample(frames[:, -CTX:], actions[:, k:k + CTX], steps, aug_bucket)
            out.append(nxt)
            frames = torch.cat([frames[:, 1:], nxt], 1)
        return torch.cat(out, 1)
