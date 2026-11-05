"""Does the world model enter a building only when the player presses into the door? (2026-11-06)

to_pewter_pc's learned policies reach the door tile (13,26) thousands of times in the real game but almost never
press up there (seed 1: never, 0/100 successes), while the imagination reward in lab/agent/imagine_rl.py pays +5 for
any imagined map change that starts on or next to the exit tile, whatever the button. This measures, from real
train-split contexts on and next to a Pokemon Center door, how often one imagined press of each direction changes the
probe-read map, next to the same rate in the real data. Control: the Viridian door, whose policies score 100/100.
Also the mats inside both Centers (leave_pewter_pc 72/100 stops on the mat; leave_viridian_pc 100/100).

Usage: python -m lab.dwm.door_warp --run wm31 --probe work/probe_ext/probe.pt --device cpu
Writes work/eval/door_warp_<run>.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lab.data import load
from lab.dwm.controllability import load_world_model
from lab.dwm.model import CTX, to_float
from lab.dwm.train import windows
from lab.probe import load_probe, predict

# (map, tiles): the exit tile(s) of the task first, then their neighbours. Doors are entered by pressing up on the door
# tile; a Pokemon Center is left by pressing down on its mat (3,7)/(4,7) (lab/agent/tasks.py).
CASES = {
    "pewter_pc door": (2, [(13, 26), (12, 26), (14, 26), (13, 27)]),
    "viridian_pc door": (1, [(23, 26), (22, 26), (24, 26), (23, 27)]),
    "pewter_pc mat": (58, [(3, 7), (4, 7), (2, 7), (5, 7), (3, 6), (4, 6)]),
    "viridian_pc mat": (41, [(3, 7), (4, 7), (2, 7), (5, 7), (3, 6), (4, 6)]),
}
NAMES = ["down", "left", "right", "up"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="wm31")
    ap.add_argument("--probe", default="work/probe_ext/probe.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--per-tile", type=int, default=32, help="real contexts per tile")
    ap.add_argument("--samples", type=int, default=4, help="imagined samples per context and button")
    ap.add_argument("--steps", type=int, default=3, help="denoising steps (imagine_rl uses 3)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data, splits = load(extended=True)
    maps, xs, ys, eps, acts = data["map_ids"], data["xs"], data["ys"], data["episode_id"], data["actions"]
    starts = windows(data, splits["train"], CTX)
    last = starts + CTX - 1
    clean = ~data["in_battles"][starts[:, None] + np.arange(CTX)].any(1)
    model = load_world_model(args.run, device=dev)
    probe = load_probe(args.probe, device=dev)
    tr = np.isin(eps, splits["train"])
    nxt_ok = np.r_[eps[1:] == eps[:-1], False] & tr  # frame t has a successor in the same train episode

    out = {"run": args.run, "probe": args.probe, "per_tile": args.per_tile, "samples": args.samples,
           "steps": args.steps, "seed": args.seed, "tiles": {}}
    for case, (m, tiles) in CASES.items():
        for x, y in tiles:
            key = f"{case} ({x},{y})"
            # real data: P(next frame is on another map | this tile, this button)
            here = np.nonzero(nxt_ok & (maps == m) & (xs == x) & (ys == y))[0]
            real = {}
            for a, name in enumerate(NAMES):
                t = here[acts[here] == a]
                real[name] = {"n": int(len(t)), "warp": float((maps[t + 1] != m).mean()) if len(t) else None}
            sel = starts[clean & (maps[last] == m) & (xs[last] == x) & (ys[last] == y)]
            row = {"real": real, "contexts": int(min(args.per_tile, len(sel))), "imagined": {}, "imagined_to": {}}
            if len(sel):
                sel = np.repeat(rng.choice(sel, min(args.per_tile, len(sel)), replace=False), args.samples)
                idx = sel[:, None] + np.arange(CTX)
                ctx = to_float(torch.from_numpy(data["frames"][idx]).to(dev))
                past = torch.from_numpy(acts[idx[:, :-1]].astype(np.int64)).to(dev)
                for a, name in enumerate(NAMES):
                    act_in = torch.cat([past, torch.full((len(sel), 1), a, dtype=torch.int64, device=dev)], 1)
                    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                        nxt = model.sample(ctx, act_in, steps=args.steps)
                    p = predict(probe, ((nxt.float() + 1) / 2).cpu().numpy().reshape(-1, 72, 80), device=dev)
                    warped = (p["map_id"] != m) & ~np.asarray(p["battle"], bool)
                    row["imagined"][name] = float(warped.mean())
                    to, n = np.unique(p["map_id"][warped], return_counts=True)  # probe-read map of each imagined change
                    row["imagined_to"][name] = {int(t): int(c) for t, c in zip(to, n)}
            out["tiles"][key] = row
            print(key, "contexts", row["contexts"], "| imagined warp:",
                  {k: round(v, 3) for k, v in row["imagined"].items()},
                  "| real warp:", {k: (v["n"], None if v["warp"] is None else round(v["warp"], 3)) for k, v in real.items()},
                  "| imagined to:", row["imagined_to"],
                  flush=True)
    path = Path("work/eval") / f"door_warp_{args.run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
