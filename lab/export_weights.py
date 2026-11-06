"""Stage the weights listed in lab/get_weights.py for upload, without optimizer state.

The loaders read only ckpt["ema"] (world model, lab/dwm/controllability.py) and ckpt["policy"] (navigation and PPO
policies, lab/agent/gym_quest.py, lab/agent/mpc.py); the optimizer state and the non-EMA weights are training-only.
Files are written to <out>/<remote path>, so the folder can be uploaded as is. Save states are skipped unless
--include-states is given (publishing them is the owner's decision).

Usage: python -m lab.export_weights --out work/hf_stage [--include-states]
"""
import argparse
import shutil
from pathlib import Path

import torch

from lab.get_weights import MANIFEST

KEEP = (("ema", ("ema", "step", "elapsed")), ("policy", ("policy", "it", "elapsed")))


def strip_checkpoint(ck):
    """Checkpoint dict -> the subset inference needs; a plain state_dict is returned unchanged."""
    for marker, keys in KEEP:
        if isinstance(ck, dict) and marker in ck:
            return {k: ck[k] for k in keys if k in ck}
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/hf_stage")
    ap.add_argument("--include-states", action="store_true")
    args = ap.parse_args()
    total_in = total_out = 0
    for name, e in MANIFEST.items():
        src, dst = Path(e["local"]), Path(args.out) / e["remote"]
        if src.suffix in (".state", ".json") and not args.include_states:
            continue
        if not src.is_file():
            print(f"missing: {name} ({src})")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".pt":
            torch.save(strip_checkpoint(torch.load(src, map_location="cpu", weights_only=True)), dst)
        else:
            shutil.copy2(src, dst)
        total_in += src.stat().st_size
        total_out += dst.stat().st_size
        print(f"{name:40s} {src.stat().st_size / 1e6:8.1f} MB -> {dst.stat().st_size / 1e6:8.1f} MB  {e['remote']}")
    print(f"total {total_in / 1e6:.1f} MB -> {total_out / 1e6:.1f} MB in {args.out}")


if __name__ == "__main__":
    main()
