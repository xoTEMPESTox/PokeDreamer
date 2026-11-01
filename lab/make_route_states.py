"""Save states along the route to the first gym, taken from real System 1/2 gym-quest runs
(lab/agent/gym_quest.py): one state at the first entry into each new map of a run, with the party it had
there. They seed exploration (lab/collect.py) past where an L8 Charmander stalls (Viridian Forest (5,0)).

Usage: python -m lab.make_route_states --seeds 100 101 102
Writes work/route_states/seed<seed>_map<id>.state and index.json (map, x, y, level, HP at each state).
"""
import argparse
import json
from pathlib import Path

import lab.env as env_mod
from lab.agent.gym_quest import run_gym_quest
from lab.agent.system12 import party_status

OUT = Path("work/route_states")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    orig_step = env_mod.PokemonRed.step
    for seed in args.seeds:
        seen, recent = set(), []

        def step(self, action, seed=seed, seen=seen, recent=recent):
            frame, ram = orig_step(self, action)
            recent[:] = (recent + [ram["map_id"]])[-3:]
            # 3 steps on the map: at warps the map ID changes 1-2 steps before x/y (agent_notes/AUDIT.md)
            if ram["map_id"] not in seen and not ram["in_battle"] and recent.count(ram["map_id"]) == 3:
                seen.add(ram["map_id"])
                path = OUT / f"seed{seed}_map{ram['map_id']}.state"
                path.write_bytes(self.snapshot())
                st = party_status(self)
                index.append({"file": str(path), "seed": seed, "map_id": ram["map_id"], "x": ram["x"], "y": ram["y"],
                              "level": st["level"], "hp": st["hp"], "max_hp": st["max_hp"]})
            return frame, ram

        env_mod.PokemonRed.step = step
        try:
            res = run_gym_quest(target_level=14, budget=18000, seed=seed, work_dir=OUT / "runs")
        finally:
            env_mod.PokemonRed.step = orig_step
        print(f"seed {seed}: badge {res['badge_obtained']}, maps {sorted(seen)}")
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} states -> {OUT}")


if __name__ == "__main__":
    main()
