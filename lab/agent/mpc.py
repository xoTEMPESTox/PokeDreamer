"""System 1 proof of concept: act in the real game by planning inside the diffusion world model.

Each real step: imagine N candidate button sequences of length H from the last 4 real frames (world model),
read the imagined map/tile with the frame probe, score progress toward the task goal, press the first button
of the best sequence in the emulator, repeat (receding horizon). Success is judged by the emulator's RAM.

Tasks (start state, exit tiles, success map) are in lab/agent/tasks.py; default "leave_house" (bedroom -> Pallet Town).
During wild battles every policy presses A (a scripted reflex, recorded per episode).

Usage: python -m lab.agent.mpc --task route1 --episodes 20 [--policy mpc|random|greedy|learned] [--video]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lab.agent.tasks import TASKS, exit_distance, train_geodesic_fields
from lab.dwm.controllability import load_world_model
from lab.dwm.model import CTX, to_float
from lab.env import PokemonRed
from lab.probe import load_probe, predict

EXITS = TASKS["leave_house"]["exits"]  # default task: bedroom stairs, 1st-floor front door
DIRECTIONS = np.array([0, 1, 2, 3])     # down, left, right, up
A, B = 4, 5


def sample_sequences(rng, n, h):
    """n button sequences of length h made of runs of 1-4 repeats of one direction."""
    out = np.empty((n, h), np.int64)
    for i in range(n):
        t = 0
        while t < h:
            run = rng.integers(1, 5)
            out[i, t:t + run] = rng.choice(DIRECTIONS)
            t += run
    return out


def score(maps, xs, ys, cur_map, cur_xy, exits=EXITS):
    """Progress toward the current map's exit, over the part of each imagined trajectory (maps/xs/ys (N, H),
    probe readings) that stays on the current real map. An imagined map change earns credit only right next to
    the exit tile, where a real warp can happen. The first version rewarded any imagined map change, and the
    planner pressed into walls chasing warps the model hallucinates (work/videos/leave_house_mpc_naive_scoring_ep0.mp4)."""
    dist = exit_distance(np.full(maps.shape, cur_map), xs, ys, exits)
    cur_d = exit_distance([cur_map], [cur_xy[0]], [cur_xy[1]], exits)[0]
    same = np.cumprod(maps == cur_map, axis=1).astype(bool)
    value = np.where(same, -dist - 0.01 * np.arange(maps.shape[1]), -np.inf).max(1)
    first_off = np.argmin(same, axis=1)
    before = np.where(first_off > 0, dist[np.arange(len(dist)), np.maximum(first_off - 1, 0)], cur_d)
    value = np.where(~same.all(1) & (before <= 1), 10.0, value)
    return np.where(np.isinf(value), -cur_d - 5.0, value)


@torch.no_grad()
def plan(model, probe, frames, past_actions, ram, rng, n, h, device, exits=EXITS):
    """Index of the best first button and the imagined frames of the best sequence."""
    seqs = sample_sequences(rng, n, h)
    ctx = to_float(torch.from_numpy(np.stack(frames[-CTX:])).to(device))[None].expand(n, -1, -1, -1)
    acts = torch.from_numpy(np.concatenate([np.tile(past_actions[-(CTX - 1):], (n, 1)), seqs], 1)).to(device)
    with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
        imagined = model.rollout(ctx, acts, steps=3)
    imagined01 = ((imagined.float() + 1) / 2).cpu().numpy()
    p = predict(probe, imagined01.reshape(-1, 72, 80), device=device)
    best = int(np.argmax(score(p["map_id"].reshape(n, h), p["x"].reshape(n, h), p["y"].reshape(n, h),
                               ram["map_id"], (ram["x"], ram["y"]), exits)))
    return int(seqs[best, 0]), imagined01[best]


def greedy_action(ram, prev_ram, prev_a, rng, exits=EXITS, fields=None):
    """Model-free control with privileged RAM: step toward the current map's exit (with `fields`: to a neighbour
    tile with a smaller path distance); after a press that did not move the player (furniture, wall), try a
    different random direction."""
    x, y = ram["x"], ram["y"]
    if fields is not None and ram["map_id"] in fields:
        f = fields[ram["map_id"]]
        at = lambda tx, ty: f[min(max(ty, 0), f.shape[0] - 1), min(max(tx, 0), f.shape[1] - 1)]
        toward = [a for a, (dx, dy) in ((0, (0, 1)), (1, (-1, 0)), (2, (1, 0)), (3, (0, -1))) if at(x + dx, y + dy) < at(x, y)]
    else:
        ex, ey = min(exits[ram["map_id"]], key=lambda t: abs(t[0] - x) + abs(t[1] - y))
        toward = [a for a, ok in ((2, ex > x), (1, ex < x), (0, ey > y), (3, ey < y)) if ok]
    blocked = prev_ram is not None and (prev_ram["map_id"], prev_ram["x"], prev_ram["y"]) == (ram["map_id"], x, y)
    if blocked:
        return int(rng.choice([a for a in DIRECTIONS if a != prev_a]))
    return int(rng.choice(toward)) if toward else int(rng.choice(DIRECTIONS))


@torch.no_grad()
def learned_action(net, frames, device):
    """Sample from the imagination-trained policy given only the last 4 real frames (no RAM, no planning)."""
    logits, _ = net(to_float(torch.from_numpy(np.stack(frames[-CTX:])).to(device))[None])
    return int(DIRECTIONS[torch.distributions.Categorical(logits=logits[0]).sample().item()])


def run_episode(policy, model, probe, rng, max_steps, n, h, device, record=False, net=None, task="leave_house",
                fields=None):
    spec = TASKS[task]
    exits = spec["exits"]
    env = PokemonRed(state=spec["state"])
    frames, actions, trace, video = [], [], [], []
    for i in range(CTX):          # fill the context with B presses (no effect in the overworld; 2.7% of training
        frame, ram = env.step(B)  # actions); skips the stale frame right after a state load (red tint in the data)
        frames.append(frame)
        if i:
            actions.append(B)
    t0 = time.time()
    success_step, prev_ram, battle_presses = None, None, 0
    for t in range(max_steps):
        if ram["in_battle"]:
            a, dream = A, None                   # scripted battle reflex for every policy
            battle_presses += 1
        elif policy == "mpc":
            a, dream = plan(model, probe, frames, actions, ram, rng, n, h, device, exits)
        elif policy == "greedy":
            a, dream = greedy_action(ram, prev_ram, actions[-1], rng, exits, fields), None
        elif policy == "learned":
            a, dream = learned_action(net, frames, device), None
        else:
            a, dream = int(rng.choice(DIRECTIONS)), None
        prev_ram = ram
        frame, ram = env.step(a)
        frames.append(frame)
        actions.append(a)
        trace.append([int(ram["map_id"]), int(ram["x"]), int(ram["y"]), a])
        if record:
            video.append((frames[-2], dream, a))
        if ram["map_id"] == spec["success_map"]:
            success_step = t + 1
            break
    env.close()
    return {"success": success_step is not None, "steps": success_step, "trace": trace,
            "battle_presses": battle_presses, "seconds": round(time.time() - t0, 1)}, video


def write_video(video, path):
    """Real frame before each decision | the agent's best imagined next 4 frames, with the chosen button."""
    names = ["down", "left", "right", "up", "A (battle)"]
    writer = None
    for real, dream, a in video:
        row = [real] + ([(d * 255).astype(np.uint8) for d in dream[:4]] if dream is not None else [])
        row = np.concatenate([np.pad(f, ((0, 0), (0, 2)), constant_values=128) for f in row], 1)
        img = cv2.cvtColor(cv2.resize(row, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        bar = np.zeros((28, img.shape[1], 3), np.uint8)
        cv2.putText(bar, f"real (left) | imagined next 4 steps | pressed: {names[a]}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        img = np.concatenate([bar, img], 0)
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 3, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer is not None:
        writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="leave_house")
    ap.add_argument("--policy", choices=["mpc", "random", "greedy", "learned"], default="mpc")
    ap.add_argument("--rl-name", default="leave_house", help="work/agent_rl/<name>/policy.pt for --policy learned")
    ap.add_argument("--run", default="main")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--n", type=int, default=64, help="candidate sequences per decision")
    ap.add_argument("--h", type=int, default=4, help="imagination horizon (steps; position is reliable to ~4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video", action="store_true", help="record the first episode")
    ap.add_argument("--geodesic", action="store_true", help="greedy: follow path distance over the train split's moves")
    args = ap.parse_args()

    model = load_world_model(args.run, device=args.device) if args.policy == "mpc" else None
    probe = load_probe(device=args.device) if args.policy == "mpc" else None
    net = None
    if args.policy == "learned":
        from lab.agent.imagine_rl import Policy
        net = Policy().to(args.device).eval()
        net.load_state_dict(torch.load(f"work/agent_rl/{args.rl_name}/policy.pt", map_location=args.device,
                                       weights_only=True)["policy"])
    fields = train_geodesic_fields(TASKS[args.task]["exits"]) if args.geodesic else None
    rng = np.random.default_rng(args.seed)
    tag = {"mpc": f"_{args.run}_n{args.n}_h{args.h}", "learned": f"_{args.rl_name}"}.get(args.policy, "") + ("_geo" if args.geodesic else "")
    stem = f"{args.task}_{args.policy}{tag}_s{args.seed}"
    results = []
    for ep in range(args.episodes):
        res, video = run_episode(args.policy, model, probe, rng, args.max_steps, args.n, args.h, args.device,
                                 record=args.video and ep == 0, net=net, task=args.task,
                                 fields=fields)
        results.append(res)
        print(f"episode {ep}: success {res['success']} steps {res['steps']} final {res['trace'][-1][:3]} "
              f"({res['seconds']} s)", flush=True)
        if video:
            Path("work/videos").mkdir(parents=True, exist_ok=True)
            write_video(video, f"work/videos/{stem}_ep0.mp4")
        k = sum(r["success"] for r in results)                # rewritten after every episode: crash-safe
        steps = [r["steps"] for r in results if r["success"]]
        summary = {"task": args.task, "policy": args.policy, "args": vars(args), "episodes": len(results),
                   "successes": k, "success_rate": k / len(results),
                   "median_steps_when_successful": float(np.median(steps)) if steps else None, "results": results}
        Path("work/agent").mkdir(parents=True, exist_ok=True)
        with open(f"work/agent/{stem}.json", "w") as fh:
            json.dump(summary, fh)
    print(f"{stem}: {k}/{len(results)} successes; median steps {summary['median_steps_when_successful']}")


if __name__ == "__main__":
    main()
