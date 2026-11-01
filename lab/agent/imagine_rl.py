"""System 1 trained purely in imagination: actor-critic on rollouts of the diffusion world model, then
deployed zero-shot in the real game (no planning at test time).

Training: start from real 4-frame contexts in the bedroom (map 38) and 1st floor (map 37) of the train
split; the policy picks buttons, the world model imagines the next frames, the frame probe reads the
imagined map/tile, and the reward is progress toward the current map's exit (same safeguard as the
planner in lab/agent/mpc.py: an imagined map change ends the rollout and is rewarded only if it happens
next to the exit tile; elsewhere it is a hallucinated warp and is penalised). Advantage actor-critic with
lambda-returns over H imagined steps (REINFORCE through the policy only; the world model is frozen).

Usage: python -m lab.agent.imagine_rl --minutes 25        (train)
       python -m lab.agent.mpc --policy learned --episodes 50   (real-game evaluation)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.agent.mpc import DIRECTIONS, EXIT
from lab.data import load
from lab.dwm.controllability import load_world_model
from lab.dwm.model import CTX, to_float
from lab.dwm.train import PAUSE, windows
from lab.probe import load_probe, predict

H, BATCH, GAMMA, LAMBDA, ENT = 10, 128, 0.95, 0.9, 0.03
STEP_COST, WARP_BONUS, HALLUCINATION_PENALTY = 0.05, 5.0, 2.0


class Policy(nn.Module):
    """Actor-critic on the last 4 frames (72x80 grey, uint8 -> [-1,1]); 4 actions (down, left, right, up)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(CTX, 32, 4, 2, 1), nn.SiLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
                                  nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(), nn.Flatten(), nn.Linear(64 * 9 * 10, 256),
                                  nn.SiLU())
        self.pi = nn.Linear(256, len(DIRECTIONS))
        self.v = nn.Linear(256, 1)

    def forward(self, frames):
        h = self.conv(frames)
        return self.pi(h), self.v(h).squeeze(-1)


def rewards_from_probe(maps, xs, ys, start_map, start_x, start_y):
    """Per-step reward, alive mask (step belongs to the rollout) and continuation mask (rollout goes on after
    the step) for imagined trajectories (B, H) starting on start_map at the real tile (start_x, start_y)."""
    b, h = maps.shape
    ex = np.array([EXIT[m][0] for m in start_map])[:, None]
    ey = np.array([EXIT[m][1] for m in start_map])[:, None]
    dist = np.abs(xs - ex) + np.abs(ys - ey)
    dist0 = np.abs(start_x - ex[:, 0]) + np.abs(start_y - ey[:, 0])
    on_map = np.cumprod(maps == start_map[:, None], axis=1).astype(bool)
    r = np.zeros((b, h), np.float32)
    alive = np.zeros((b, h), np.float32)
    for k in range(h):
        prev = dist[:, k - 1] if k else dist0
        left = ~on_map[:, k] & (on_map[:, k - 1] if k else True)
        r[:, k] = np.where(on_map[:, k], prev - dist[:, k] - STEP_COST, 0.0)
        r[:, k] = np.where(left, np.where(prev <= 1, WARP_BONUS, -HALLUCINATION_PENALTY), r[:, k])
        alive[:, k] = on_map[:, k] | left
    return r, alive, on_map.astype(np.float32)


def lambda_returns(r, v, cont, bootstrap):
    """TD(lambda) targets; r, v (= V(s_k)), cont (B, H) with cont=0 where the rollout ends at step k;
    bootstrap (B,) = V(s_H) after the last imagined step."""
    out = torch.zeros_like(r)
    nxt = bootstrap
    for k in reversed(range(r.shape[1])):
        nv = v[:, k + 1] if k + 1 < r.shape[1] else bootstrap
        nxt = r[:, k] + GAMMA * cont[:, k] * ((1 - LAMBDA) * nv + LAMBDA * nxt)
        out[:, k] = nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--name", default="leave_house")
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    out = Path("work/agent_rl") / args.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    data, splits = load()
    starts = windows(data, splits["train"], CTX)
    last = starts + CTX - 1
    ok = np.isin(data["map_ids"][last], list(EXIT)) & ~data["in_battles"][starts[:, None] + np.arange(CTX)].any(1)
    starts = starts[ok]
    log(f"{len(starts)} real start contexts (maps {sorted(set(data['map_ids'][starts + CTX - 1].tolist()))})")
    frames = torch.from_numpy(data["frames"]).to(dev)
    acts = torch.from_numpy(data["actions"].astype(np.int64)).to(dev)

    model = load_world_model(args.run, device=dev)
    probe = load_probe(device=dev)
    policy = Policy().to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    ckpt = out / "policy.pt"
    it, elapsed = 0, 0.0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=dev, weights_only=True)
        policy.load_state_dict(st["policy"])
        opt.load_state_dict(st["opt"])
        it, elapsed = st["it"], st["elapsed"]
        log(f"resumed at iteration {it}")
    rng = np.random.default_rng(it)
    ar = torch.arange(CTX, device=dev)
    t_last = time.time()
    # equal starts per map: the 1st floor is ~1.4% of contexts, and an unbalanced run (to it 220) learned the
    # bedroom stairs (30/30 real episodes reached the 1st floor) but walked straight back up from there
    by_map = [starts[data["map_ids"][starts + CTX - 1] == m] for m in EXIT]
    log("start contexts per map: " + ", ".join(f"{m}: {len(b)}" for m, b in zip(EXIT, by_map)))
    while elapsed < args.minutes * 60:
        s = torch.from_numpy(np.concatenate([rng.choice(b, BATCH // len(by_map)) for b in by_map])).to(dev)
        ctx = to_float(frames[s[:, None] + ar])                       # (B, CTX, H, W)
        past = acts[s[:, None] + ar[:-1]]                             # actions between the context frames
        last_i = (s + CTX - 1).cpu().numpy()
        start_map, start_x, start_y = (data[k][last_i].astype(np.int64) for k in ("map_ids", "xs", "ys"))
        obs, chosen, imagined = [], [], []
        with torch.no_grad():
            for k in range(H):
                logits, _ = policy(ctx)
                a = torch.distributions.Categorical(logits=logits).sample()
                obs.append(ctx)
                chosen.append(a)
                act_in = torch.cat([past, DIRECTIONS_T.to(dev)[a][:, None]], 1)
                with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                    nxt = model.sample(ctx, act_in, steps=3)
                imagined.append(nxt.float())
                ctx = torch.cat([ctx[:, 1:], nxt.float()], 1)
                past = act_in[:, 1:]
            imag = torch.cat(imagined, 1)                             # (B, H, 72, 80)
            p = predict(probe, ((imag + 1) / 2).cpu().numpy().reshape(-1, 72, 80), device=dev)
            r, alive, cont = rewards_from_probe(p["map_id"].reshape(BATCH, H), p["x"].reshape(BATCH, H),
                                                p["y"].reshape(BATCH, H), start_map, start_x, start_y)
            _, v_last = policy(ctx)
        obs_t = torch.stack(obs, 1)                                   # (B, H, CTX, 72, 80)
        logits, v = policy(obs_t.flatten(0, 1))
        logits, v = logits.view(BATCH, H, -1), v.view(BATCH, H)
        r_t, alive_t, cont_t = (torch.from_numpy(x).to(dev) for x in (r, alive, cont))
        target = lambda_returns(r_t, v.detach(), cont_t, v_last.detach())
        adv = (target - v.detach())
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        logp = F.log_softmax(logits, -1)
        a_t = torch.stack(chosen, 1)
        pg = -(logp.gather(-1, a_t[..., None]).squeeze(-1) * adv * alive_t).sum() / alive_t.sum()
        vloss = ((v - target) ** 2 * alive_t).sum() / alive_t.sum()
        ent = -(logp.exp() * logp).sum(-1).mul(alive_t).sum() / alive_t.sum()
        loss = pg + 0.5 * vloss - ENT * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        it += 1
        if it % 20 == 0:
            now = time.time()
            elapsed += now - t_last
            ret = (r * alive).sum(1).mean()
            hall = (r == -HALLUCINATION_PENALTY).any(1).mean()
            warp = (r == WARP_BONUS).any(1).mean()
            log(f"it {it} imagined return {ret:.3f} warp-through-exit {warp:.3f} hallucinated-warp {hall:.3f} "
                f"entropy {ent.item():.3f} value-loss {vloss.item():.3f} ({20 / (now - t_last):.2f} it/s, "
                f"{elapsed / 60:.1f} min)")
            tmp = out / "policy.tmp"
            torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(), "it": it, "elapsed": elapsed}, tmp)
            os.replace(tmp, ckpt)
            if PAUSE.exists():
                log("paused (work/PAUSE exists)")
                while PAUSE.exists():
                    time.sleep(10)
                log("resumed")
            t_last = time.time()
    log(f"DONE {it} iterations, {elapsed / 60:.1f} min; policy {ckpt}")
    with open(out / "args.json", "w") as fh:
        json.dump(vars(args), fh)


DIRECTIONS_T = torch.from_numpy(DIRECTIONS.astype(np.int64))

if __name__ == "__main__":
    main()
