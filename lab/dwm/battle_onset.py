"""Does the world model imagine wild encounters? Open-loop rollouts from real contexts (held-out
episodes, chosen map, no battle in the context) driven by the real future actions. Compares how often a battle
starts within H steps in RAM, in the probe's reading of the real frames, and in the probe's reading of the
imagined frames (several samples per context: the sampler is deterministic given its initial noise).

Usage: python -m lab.dwm.battle_onset --run main --map 12
       python -m lab.dwm.battle_onset --run wm31 --data ext --map 12
Writes work/eval/battle_onset_<run>_map<map>[_ext].json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--map", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--split", default=None, help="split to evaluate (default: test_ext for ext, test_id for base)")
    ap.add_argument("--data", choices=["base", "ext"], default="base", help="base or extended dataset")
    ap.add_argument("--probe", default=None, help="path to probe checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3, help="Euler sampling steps (DIAMOND default 3)")
    ap.add_argument("--max-windows", type=int, default=0, help="max windows to evaluate (0 = all)")
    ap.add_argument("--tag", default="", help="optional tag appended to output filename")
    args = ap.parse_args()
    H, dev = args.horizon, args.device

    split = args.split or ("test_ext" if args.data == "ext" else "test_id")
    probe_path = args.probe or ("work/probe_ext/probe.pt" if args.data == "ext" else "work/probe/probe.pt")

    data, splits = load(extended=args.data == "ext")
    starts = windows(data, splits[split], CTX + H)
    ok = (data["map_ids"][starts + CTX - 1] == args.map) & ~data["in_battles"][starts[:, None] + np.arange(CTX)].any(1)
    starts = starts[ok]
    if args.max_windows and len(starts) > args.max_windows:
        starts = starts[:args.max_windows]
    fut = starts[:, None] + CTX + np.arange(H)
    ram = data["in_battles"][fut].astype(bool)

    model = load_world_model(args.run, device=dev)
    probe = load_probe(probe_path, device=dev)
    real_probe = predict(probe, data["frames"][fut].reshape(-1, 72, 80), device=dev)["battle"].reshape(-1, H)
    imag = []
    for rep in range(args.samples):
        torch.manual_seed(rep)
        out = []
        for i in range(0, len(starts), 256):
            s = starts[i:i + 256]
            ctx = to_float(torch.from_numpy(data["frames"][s[:, None] + np.arange(CTX)]).to(dev))
            act = torch.from_numpy(data["actions"][s[:, None] + np.arange(CTX + H - 1)]).long().to(dev)
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                frames = model.rollout(ctx, act, steps=args.steps)
            out.append(((frames.float() + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy())
        frames = np.concatenate(out)
        imag.append(predict(probe, frames.reshape(-1, 72, 80), device=dev)["battle"].reshape(-1, H))
    imag = np.stack(imag)                                             # (samples, windows, H)

    res = {"run": args.run, "map": args.map, "split": split, "horizon": H, "windows": int(len(starts)),
           "episodes": int(len(set(data["episode_id"][starts].tolist()))), "samples": args.samples, "sampling_steps": args.steps,
           "battle_within_H": {"ram": float(ram.any(1).mean()), "probe_on_real_frames": float(real_probe.any(1).mean()),
                               "probe_on_imagined_frames": float(imag.any(2).mean()),
                               "imagined_any_of_samples": float(imag.any((0, 2)).mean())},
           "battle_frame_fraction": {"ram": float(ram.mean()), "probe_on_real_frames": float(real_probe.mean()),
                                     "probe_on_imagined_frames": float(imag.mean())},
           "imagined_given_real_onset": float(imag.any(2)[:, ram.any(1)].mean()) if ram.any() else None}
    if args.data != "base":
        res["data"] = args.data
    suffix = ("_ext" if args.data == "ext" else "") + (f"_{args.tag}" if args.tag else "")
    out = Path("work/eval") / (f"battle_onset_{args.run}_map{args.map}" + (f"_s{args.steps}" if args.steps != 3 else "") + f"{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
