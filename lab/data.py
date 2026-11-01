"""Build the 72x80 greyscale cache, the episode-level splits and dataset statistics.

Usage: python -m lab.data   (from the repo root)
Writes work/cache_72x80.npz, work/splits.json, agent_notes/results/dataset_stats.md
"""
import collections
import glob
import json
from pathlib import Path

import cv2
import numpy as np

SRC = "work/hf/dataset/transitions_*.npz"
CACHE = Path("work/cache_72x80.npz")
SPLITS = Path("work/splits.json")
H, W = 72, 80          # exact 2x downsample of 144x160 keeps the 8-px tile grid (4 px per tile)
OOD_START_MAP = 0      # hold out every episode that starts in Pallet Town
SEED = 0
RAM_KEYS = ["actions", "episode_starts", "map_ids", "xs", "ys", "in_battles", "facings"]


def to_small(obs):
    """(N,144,160,3) uint8 RGB -> (N,72,80) uint8 grey, area-averaged."""
    out = np.empty((len(obs), H, W), np.uint8)
    for i, f in enumerate(obs):
        out[i] = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (W, H), interpolation=cv2.INTER_AREA)
    return out


def build_cache():
    files = sorted(glob.glob(SRC))
    frames, ram = [], collections.defaultdict(list)
    for f in files:
        d = np.load(f)
        frames.append(to_small(d["obs"]))
        for k in RAM_KEYS:
            ram[k].append(d[k])
    data = {k: np.concatenate(v) for k, v in ram.items()}
    data["frames"] = np.concatenate(frames)
    data["episode_id"] = np.cumsum(data["episode_starts"]) - 1
    np.savez(CACHE, **data)
    return data


def make_splits(data):
    starts = np.where(data["episode_starts"])[0]
    start_map = data["map_ids"][starts]
    ood = [int(e) for e in np.where(start_map == OOD_START_MAP)[0]]
    rest = np.where(start_map != OOD_START_MAP)[0]
    rng = np.random.default_rng(SEED)
    test_id = sorted(int(e) for e in rng.choice(rest, size=round(0.1 * len(rest)), replace=False))
    train = sorted(int(e) for e in rest if e not in set(test_id))
    splits = {"seed": SEED, "ood_start_map": OOD_START_MAP, "train": train, "test_id": test_id, "test_ood": ood,
              "start_map_by_episode": [int(m) for m in start_map]}
    SPLITS.write_text(json.dumps(splits))
    return splits


def stats(data, splits):
    ep = data["episode_id"]
    same_ep = ep[1:] == ep[:-1]
    moved = ((data["xs"][1:] != data["xs"][:-1]) | (data["ys"][1:] != data["ys"][:-1]))[same_ep]
    map_change = (data["map_ids"][1:] != data["map_ids"][:-1])[same_ep]
    lines = ["# Dataset statistics (measured)", "",
             "Source: `work/cache_72x80.npz` built by `lab/data.py` from the HF dataset; splits in `work/splits.json`.", ""]
    lines.append(f"- Steps: {len(ep)}; episodes: {ep.max() + 1} (all 1000 steps)")
    lines.append(f"- Start maps: {dict(collections.Counter(splits['start_map_by_episode']))}")
    lines.append(f"- Split (episodes): train {len(splits['train'])}, test_id {len(splits['test_id'])}, "
                 f"test_ood (start map {OOD_START_MAP}) {len(splits['test_ood'])}")
    lines.append(f"- Steps where (x, y) changes: {moved.mean():.3f} (also counts map transitions)")
    lines.append(f"- Steps where map ID changes: {map_change.mean():.4f}")
    lines.append(f"- Steps in battle: {data['in_battles'].mean():.3f}")
    lines.append("- `dialog_opens` is True on 95% of steps, so it is not a usable dialog flag (not used).")
    lines.append(f"- Action counts: {dict(sorted(collections.Counter(data['actions'].tolist()).items()))}")
    lines += ["", "| map ID | steps | fraction |", "|---|---|---|"]
    for m, c in collections.Counter(data["map_ids"].tolist()).most_common():
        lines.append(f"| {m} | {c} | {c / len(ep):.3f} |")
    for name in ["train", "test_id", "test_ood"]:
        mask = np.isin(ep, splits[name])
        cnt = collections.Counter(data["map_ids"][mask].tolist())
        lines.append(f"\n{name} map distribution: " + ", ".join(f"{m}: {c / mask.sum():.3f}" for m, c in cnt.most_common()))
    Path("agent_notes/results").mkdir(parents=True, exist_ok=True)
    Path("agent_notes/results/dataset_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def load():
    d = np.load(CACHE)
    return {k: d[k] for k in d.files}, json.loads(SPLITS.read_text())


if __name__ == "__main__":
    data = build_cache()
    stats(data, make_splits(data))
