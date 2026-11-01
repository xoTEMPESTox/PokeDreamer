"""System 1 proof of concept: act in the real game by planning inside the diffusion world model.

Each real step: imagine N candidate button sequences of length H from the last 4 real frames (world model),
read the imagined map/tile with the frame probe, score progress toward the task goal, press the first button
of the best sequence in the emulator, repeat (receding horizon). Success is judged by the emulator's RAM.

Task "leave_house": start in Red's bedroom (map 38), succeed on reaching Pallet Town (map 0). Exit tiles come
from the dataset's own map transitions (bedroom stairs (7,1): 27 transitions; front door (2,7): only 3).

Usage: python -m lab.agent.mpc --episodes 20 [--policy mpc|random] [--video]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lab.dwm.controllability import load_world_model
from lab.dwm.model import CTX, to_float
from lab.env import PokemonRed
from lab.eval.metrics import manhattan
from lab.probe import load_probe, predict

EXIT = {38: (7, 1), 37: (2, 7)}         # bedroom stairs, 1st-floor front door (then Pallet Town = success)
DIRECTIONS = np.array([0, 1, 2, 3])     # down, left, right, up
B = 5


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


def score(maps, xs, ys, cur_map, cur_xy):
    """Progress toward the current map's exit, over the part of each imagined trajectory (maps/xs/ys (N, H),
    probe readings) that stays on the current real map. An imagined map change earns credit only right next to
    the exit tile, where a real warp can happen. The first version rewarded any imagined map change, and the
    planner pressed into walls chasing warps the model hallucinates (work/videos/leave_house_mpc_naive_scoring_ep0.mp4)."""
    ex = np.array(EXIT[cur_map])
    dist = manhattan(np.stack([xs, ys], -1), ex)
    same = np.cumprod(maps == cur_map, axis=1).astype(bool)
    value = np.where(same, -dist - 0.01 * np.arange(maps.shape[1]), -np.inf).max(1)
    first_off = np.argmin(same, axis=1)
    before = np.where(first_off > 0, dist[np.arange(len(dist)), np.maximum(first_off - 1, 0)],
                      manhattan(np.array(cur_xy), ex))
    value = np.where(~same.all(1) & (before <= 1), 10.0, value)
    return np.where(np.isinf(value), -manhattan(np.array(cur_xy), ex) - 5.0, value)


@torch.no_grad()
def plan(model, probe, frames, past_actions, ram, rng, n, h, device):
    """Index of the best first button and the imagined frames of the best sequence."""
    seqs = sample_sequences(rng, n, h)
    ctx = to_float(torch.from_numpy(np.stack(frames[-CTX:])).to(device))[None].expand(n, -1, -1, -1)
    acts = torch.from_numpy(np.concatenate([np.tile(past_actions[-(CTX - 1):], (n, 1)), seqs], 1)).to(device)
    with torch.autocast("cuda", torch.bfloat16, enabled=device == "cuda"):
        imagined = model.rollout(ctx, acts, steps=3)
    imagined01 = ((imagined.float() + 1) / 2).cpu().numpy()
    p = predict(probe, imagined01.reshape(-1, 72, 80), device=device)
    best = int(np.argmax(score(p["map_id"].reshape(n, h), p["x"].reshape(n, h), p["y"].reshape(n, h),
                               ram["map_id"], (ram["x"], ram["y"]))))
    return int(seqs[best, 0]), imagined01[best]


def run_episode(policy, model, probe, rng, max_steps, n, h, device, record=False):
    env = PokemonRed(state="saves/bedroom.state")
    frames, actions, trace, video = [], [], [], []
    for i in range(CTX):          # fill the context with B presses (no effect in the overworld; 2.7% of training
        frame, ram = env.step(B)  # actions); skips the stale frame right after a state load (red tint in the data)
        frames.append(frame)
        if i:
            actions.append(B)
    t0 = time.time()
    success_step = None
    for t in range(max_steps):
        if policy == "mpc":
            a, dream = plan(model, probe, frames, actions, ram, rng, n, h, device)
        else:
            a, dream = int(rng.choice(DIRECTIONS)), None
        frame, ram = env.step(a)
        frames.append(frame)
        actions.append(a)
        trace.append([int(ram["map_id"]), int(ram["x"]), int(ram["y"]), a])
        if record:
            video.append((frames[-2], dream, a))
        if ram["map_id"] == 0:
            success_step = t + 1
            break
    env.close()
    return {"success": success_step is not None, "steps": success_step, "trace": trace,
            "seconds": round(time.time() - t0, 1)}, video


def write_video(video, path):
    """Real frame before each decision | the agent's best imagined next 4 frames, with the chosen button."""
    names = ["down", "left", "right", "up"]
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
    ap.add_argument("--policy", choices=["mpc", "random"], default="mpc")
    ap.add_argument("--run", default="main")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--n", type=int, default=64, help="candidate sequences per decision")
    ap.add_argument("--h", type=int, default=4, help="imagination horizon (steps; position is reliable to ~4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video", action="store_true", help="record the first episode")
    args = ap.parse_args()

    model = load_world_model(args.run, device=args.device) if args.policy == "mpc" else None
    probe = load_probe(device=args.device) if args.policy == "mpc" else None
    rng = np.random.default_rng(args.seed)
    stem = f"leave_house_{args.policy}" + (f"_{args.run}_n{args.n}_h{args.h}" if args.policy == "mpc" else "") + f"_s{args.seed}"
    results = []
    for ep in range(args.episodes):
        res, video = run_episode(args.policy, model, probe, rng, args.max_steps, args.n, args.h, args.device,
                                 record=args.video and ep == 0)
        results.append(res)
        print(f"episode {ep}: success {res['success']} steps {res['steps']} final {res['trace'][-1][:3]} "
              f"({res['seconds']} s)", flush=True)
        if video:
            Path("work/videos").mkdir(parents=True, exist_ok=True)
            write_video(video, f"work/videos/{stem}_ep0.mp4")
    k = sum(r["success"] for r in results)
    steps = [r["steps"] for r in results if r["success"]]
    summary = {"task": "leave_house", "policy": args.policy, "args": vars(args), "episodes": len(results),
               "successes": k, "success_rate": k / len(results),
               "median_steps_when_successful": float(np.median(steps)) if steps else None, "results": results}
    Path("work/agent").mkdir(parents=True, exist_ok=True)
    with open(f"work/agent/{stem}.json", "w") as fh:
        json.dump(summary, fh)
    print(f"{stem}: {k}/{len(results)} successes; median steps {summary['median_steps_when_successful']}")


if __name__ == "__main__":
    main()
