"""Record a GIF of a learned policy playing the real game (native 160x144 screen, 2x), one episode.

Usage: python -m lab.agent.record --task route1 --rl-name route1 --out work/videos/route1_policy.gif [--seed 7]
"""
import argparse

import torch
from PIL import Image, ImageDraw

from lab.agent.imagine_rl import Policy
from lab.agent.mpc import A, B, CTX, learned_action
from lab.agent.tasks import TASKS
from lab.env import PokemonRed
from src.game_state import screen_capture

NAMES = ["down", "left", "right", "up", "A (battle)"]


def panel(raw, text, color=(255, 255, 255)):
    img = Image.fromarray(raw).resize((320, 288), Image.NEAREST)
    canvas = Image.new("RGB", (320, 312), (20, 20, 20))
    canvas.paste(img, (0, 24))
    ImageDraw.Draw(canvas).text((6, 6), text, fill=color)
    return canvas.convert("P", palette=Image.ADAPTIVE, colors=32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="route1")
    ap.add_argument("--rl-name", default="route1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    spec = TASKS[args.task]
    net = Policy().eval()
    net.load_state_dict(torch.load(f"work/agent_rl/{args.rl_name}/policy.pt", map_location="cpu", weights_only=True)["policy"])
    env = PokemonRed(state=spec["state"])
    frames, shots = [], []
    for _ in range(CTX):
        frame, ram = env.step(B)
        frames.append(frame)
    for t in range(args.max_steps):
        a = A if ram["in_battle"] else learned_action(net, frames, "cpu")
        shots.append(panel(screen_capture(env.pyboy), f"real game | press {t + 1}: {NAMES[a]}"))
        frame, ram = env.step(a)
        frames.append(frame)
        if ram["map_id"] == spec["success_map"]:
            break
    for _ in range(2):
        env.step(7)                          # let the map transition finish before the last frame
    ok = ram["map_id"] == spec["success_map"]
    shots += [panel(screen_capture(env.pyboy), f"{'reached' if ok else 'not reached'} after {t + 1} presses",
                    (120, 255, 120) if ok else (255, 120, 120))] * 6
    env.close()
    shots[0].save(args.out, save_all=True, append_images=shots[1:], duration=250, loop=0, optimize=True)
    print(f"{args.out}: {t + 1} presses, success {ok}")


if __name__ == "__main__":
    main()
