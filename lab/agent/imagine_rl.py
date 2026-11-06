"""System 1 trained purely in imagination: actor-critic on rollouts of the diffusion world model, then
deployed zero-shot in the real game (no planning at test time).

Training: start from real 4-frame contexts in the bedroom (map 38) and 1st floor (map 37) of the train
split; the policy picks buttons, the world model imagines the next frames, the frame probe reads the
imagined map/tile, and the reward is progress toward the current map's exit (same safeguard as the
planner in lab/agent/mpc.py: an imagined map change ends the rollout and is rewarded only if it happens
next to the exit tile; elsewhere it is a hallucinated warp and is penalised). Advantage actor-critic with
lambda-returns over H imagined steps (REINFORCE through the policy only; the world model is frozen).

Tasks (exit tiles per map) come from lab/agent/tasks.py. If the probe reads an imagined frame as a battle, the
rollout ends with no bonus or penalty (the probe cannot read the map from a battle screen).

Usage: python -m lab.agent.imagine_rl --task route1 --name route1 --minutes 30   (train)
       python -m lab.agent.mpc --task route1 --policy learned --rl-name route1 --episodes 100   (real game)
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

from lab.agent.mpc import DIRECTIONS
from lab.agent.tasks import TASKS, exit_distance, train_geodesic_fields
from lab.data import load
from lab.dwm.controllability import load_world_model
from lab.dwm.model import CTX, to_float
from lab.dwm.train import PAUSE, windows
from lab.probe import load_probe, predict

H, BATCH, GAMMA, LAMBDA, ENT = 10, 128, 0.95, 0.9, 0.03
STEP_COST, WARP_BONUS, HALLUCINATION_PENALTY = 0.05, 5.0, 2.0
CENTERS = (41, 58)  # Viridian and Pewter Pokemon Centers (lab/agent/nav.py)


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


def rewards_from_probe(maps, xs, ys, start_map, start_x, start_y, exits=TASKS["leave_house"]["exits"], battle=None,
                       fields=None, step_clip=None, move_gate=None, warp_dist=1, merge_centers=False):
    """Per-step reward, alive mask (step belongs to the rollout) and continuation mask (rollout goes on after
    the step) for imagined trajectories (B, H) starting on start_map at the real tile (start_x, start_y).
    step_clip bounds the per-step progress term: a press moves at most one tile, so a larger change in the
    probe-read distance is a misread or a hallucinated jump (viridian_geo hit -30 single-step rewards).
    Clipping is not potential-based: a closed loop with one big real drop (a ledge jump, 26 -> 31 -> 30) earns
    +4 (the to_route2 trap, 2026-11-04). move_gate instead accepts a probe-read tile only if it is within
    move_gate tiles (Manhattan) of the last accepted tile and has an observed path to the exit; progress is
    measured from the last accepted tile and rejected reads earn none, so rewards telescope (as in ppo_real).
    An imagined map change earns WARP_BONUS if it leaves from within warp_dist of the exit, else the hallucination
    penalty. warp_dist 0 pays only changes from the exit tile itself: wm31 imagines the Pewter PC after presses beside
    its door that never enter it in the real game (work/eval/door_warp_wm31.json, 2026-11-06). merge_centers reads
    another Pokemon Center as the start Center: they share one interior, and the probe reads wm31's Pewter PC frames
    after a right press on the mat as the Viridian PC (same file)."""
    b, h = maps.shape
    battle = np.zeros((b, h), bool) if battle is None else np.asarray(battle, bool)
    if merge_centers:
        maps = np.where(np.isin(maps, CENTERS) & np.isin(start_map, CENTERS)[:, None], start_map[:, None], maps)
    dist = exit_distance(np.broadcast_to(start_map[:, None], (b, h)), xs, ys, exits, fields)
    dist0 = exit_distance(start_map, start_x, start_y, exits, fields)
    on_map = np.cumprod((maps == start_map[:, None]) & ~battle, axis=1).astype(bool)
    r = np.zeros((b, h), np.float32)
    alive = np.zeros((b, h), np.float32)
    if move_gate is not None:  # last accepted tile; unobserved tiles of a geodesic field hold its max
        ref_x, ref_y, ref_d = np.asarray(start_x), np.asarray(start_y), dist0
        top = np.array([fields[m].max() if fields is not None and m in fields else np.inf for m in start_map])
    for k in range(h):
        prev = dist[:, k - 1] if k else dist0
        left = ~on_map[:, k] & (on_map[:, k - 1] if k else True)
        if move_gate is not None:
            ok = (np.abs(xs[:, k] - ref_x) + np.abs(ys[:, k] - ref_y) <= move_gate) & (dist[:, k] < top)
            prev, progress = ref_d, np.where(ok, ref_d - dist[:, k], 0.0)
            ref_x, ref_y, ref_d = np.where(ok, xs[:, k], ref_x), np.where(ok, ys[:, k], ref_y), np.where(ok, dist[:, k], ref_d)
        else:
            progress = prev - dist[:, k] if step_clip is None else np.clip(prev - dist[:, k], -step_clip, step_clip)
        r[:, k] = np.where(on_map[:, k], progress - STEP_COST, 0.0)
        r[:, k] = np.where(left, np.where(battle[:, k], -STEP_COST, np.where(prev <= warp_dist, WARP_BONUS, -HALLUCINATION_PENALTY)), r[:, k])
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
    ap.add_argument("--task", choices=sorted(TASKS), default="leave_house")
    ap.add_argument("--name", default="leave_house")
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--max-iters", type=int, default=0,
                    help="also stop at this iteration (0 = no limit): matches runs made at another GPU power limit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe", default="work/probe/probe.pt", help="path to probe checkpoint")
    ap.add_argument("--data", choices=["base", "ext"], default="base", help="base or extended dataset")
    ap.add_argument("--geodesic", action="store_true",
                    help="reward progress in path distance over the train split's observed moves, not Manhattan")
    ap.add_argument("--step-clip", type=float, default=None, help="clip per-step progress to +-this many tiles")
    ap.add_argument("--move-gate", type=float, default=None,
                    help="instead of --step-clip: count progress only from reads within this many tiles of the last accepted one")
    ap.add_argument("--warp-dist", type=float, default=1,
                    help="an imagined map change earns the warp bonus only from within this distance of the exit (0: the exit tile)")
    ap.add_argument("--ent", type=float, default=ENT, help=f"entropy bonus coefficient (default {ENT})")
    ap.add_argument("--merge-centers", action="store_true",
                    help="read another Pokemon Center as the start Center (one shared interior; the probe confuses them)")
    ap.add_argument("--seed", type=int, default=None, help="seed torch and the start sampling (default: unseeded)")
    args = ap.parse_args()
    if args.step_clip is not None and args.move_gate is not None:
        ap.error("--step-clip and --move-gate are alternatives")
    dev = args.device
    out = Path("work/agent_rl") / args.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    data, splits = load(extended=args.data == "ext")
    starts = windows(data, splits["train"], CTX)
    last = starts + CTX - 1
    exits = TASKS[args.task]["exits"]
    fields = train_geodesic_fields(exits, extended=args.data == "ext") if args.geodesic else None
    log(f"distance to exit: {'geodesic (observed moves, train split)' if args.geodesic else 'Manhattan'}; "
        f"step clip {args.step_clip}; move gate {args.move_gate}; warp dist {args.warp_dist}; merge centers {args.merge_centers}; "
        f"entropy coef {args.ent}")
    train_maps = TASKS[args.task].get("train_maps", list(exits))
    ok = np.isin(data["map_ids"][last], train_maps) & ~data["in_battles"][starts[:, None] + np.arange(CTX)].any(1)
    if fields is not None:  # only starts with an observed path to the exit (to_pewter: 76% of Route 2 is south of the forest)
        lm = data["map_ids"][last]
        top = np.full(len(last), np.inf)
        for m, f in fields.items():
            top[lm == m] = f.max()
        ok &= exit_distance(lm, data["xs"][last], data["ys"][last], exits, fields) < top
    starts = starts[ok]
    log(f"{len(starts)} real start contexts (maps {sorted(set(data['map_ids'][starts + CTX - 1].tolist()))})")
    frames = torch.from_numpy(data["frames"]).to(dev)
    acts = torch.from_numpy(data["actions"].astype(np.int64)).to(dev)

    model = load_world_model(args.run, device=dev)
    probe = load_probe(args.probe, device=dev)
    if args.seed is not None:  # policy init and sampling; GPU kernels are not bit-deterministic
        torch.manual_seed(args.seed)
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
    rng = np.random.default_rng(it if args.seed is None else [args.seed, it])
    ar = torch.arange(CTX, device=dev)
    t_last = time.time()
    # equal starts per map: the 1st floor is ~1.4% of contexts, and an unbalanced run (to it 220) learned the
    # bedroom stairs (30/30 real episodes reached the 1st floor) but walked straight back up from there
    by_map = [starts[data["map_ids"][starts + CTX - 1] == m] for m in train_maps]
    log(f"task {args.task}; start contexts per map: " + ", ".join(f"{m}: {len(b)}" for m, b in zip(train_maps, by_map)))
    while elapsed < args.minutes * 60 and not (args.max_iters and it >= args.max_iters):
        counts = [BATCH // len(by_map) + (i < BATCH % len(by_map)) for i in range(len(by_map))]  # sums to BATCH
        s = torch.from_numpy(np.concatenate([rng.choice(b, c) for b, c in zip(by_map, counts)])).to(dev)
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
            battle = p["battle"].reshape(BATCH, H)
            r, alive, cont = rewards_from_probe(p["map_id"].reshape(BATCH, H), p["x"].reshape(BATCH, H),
                                                p["y"].reshape(BATCH, H), start_map, start_x, start_y, exits, battle, fields,
                                                args.step_clip, args.move_gate, args.warp_dist, args.merge_centers)
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
        loss = pg + 0.5 * vloss - args.ent * ent
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
            bat = (battle & (alive > 0)).any(1).mean()
            warp = (r == WARP_BONUS).any(1).mean()
            log(f"it {it} imagined return {ret:.3f} warp-through-exit {warp:.3f} hallucinated-warp {hall:.3f} battle {bat:.3f} "
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
