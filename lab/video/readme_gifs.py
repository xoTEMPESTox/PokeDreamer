"""GIFs for the README (and the website) from the recorded video run.

  gameplay   stage-1 colour frames of the real run (Viridian to the Route 1 grass), 2x
  dream      real colour | the world model's imagination (stage-2 colour), side by side
  opener     the 3D pond opener (work/viz3d/opener_8s.mp4), downscaled

Inputs (work/, not tracked): work/video/traj/run_seed_100{,_color}.npz, work/video/dream_4_8718_k10.npz,
work/viz3d/opener_8s.mp4, work/color/learned/model.pt, work/upsampler/upsampler.pt.
Usage: python -m lab.video.readme_gifs [--out docs/assets/v3]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def save_gif(frames, path, fps):
    imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"{path}: {len(frames)} frames, {Path(path).stat().st_size / 1e6:.2f} MB")


def nearest(img, k):
    return np.kron(img, np.ones((k, k, 1), np.uint8))


def gameplay(out, start=78, end=318, stride=2, fps=15):
    color = np.load("work/video/traj/run_seed_100_color.npz")["color"]
    save_gif([nearest(color[s], 2) for s in range(start, end, stride)], out / "gameplay_colour.gif", fps)


def dream(out, start=7200, end=7320, stride=2, fps=12):
    from lab.color.learned import colorize72, load_colorizer, load_palette
    from lab.upsampler import load_upsampler
    d = np.load("work/video/dream_4_8718_k10.npz")
    color = np.load("work/video/traj/run_seed_100_color.npz")["color"]
    off = int(d["start"])
    steps = list(range(start, end, stride))
    dream_rgb = colorize72(d["dream"][[s - off for s in steps]], load_colorizer("work/color/learned/model.pt"),
                           load_upsampler("work/upsampler/upsampler.pt"), load_palette())
    gap = np.full((144, 6, 3), 255, np.uint8)
    frames = [nearest(np.concatenate([color[s], gap, dream_rgb[i]], 1), 2) for i, s in enumerate(steps)]
    save_gif(frames, out / "real_vs_dream.gif", fps)


def opener(out, every=2, fps=15, width=640):
    cap = cv2.VideoCapture("work/viz3d/opener_8s.mp4")
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            h = int(f.shape[0] * width / f.shape[1])
            frames.append(cv2.cvtColor(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        i += 1
    save_gif(frames, out / "opener_3d.gif", fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets/v3")
    ap.add_argument("--only", choices=["gameplay", "dream", "opener"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in [("gameplay", gameplay), ("dream", dream), ("opener", opener)]:
        if args.only in (None, name):
            fn(out)


if __name__ == "__main__":
    main()
