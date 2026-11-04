"""tile_sweep.py — Start-tile robustness sweep for leg policies in PokeDreamer.

Measures how policy success depends on the start tile for navigation legs.
Extracts reachable tiles stratified by BFS distance to the map exit,
creates emulator start states by walking from the task's start state,
evaluates policies in parallel CPU subprocesses calling lab.agent.mpc,
and aggregates results into summary.csv and table.md.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from lab.agent.nav import build_walkability_grid, navigate_to
from lab.agent.tasks import TASKS
from lab.env import PokemonRed


def _make_walkable_fn(is_walkable: Callable[[int, int], bool] | Any) -> Callable[[int, int], bool]:
    """Normalize callable or 2D grid/array into a (x, y) -> bool function."""
    if callable(is_walkable):
        return is_walkable
    return lambda x, y: bool(is_walkable[y][x])


def compute_exit_distances(
    is_walkable: Callable[[int, int], bool] | Any,
    w: int,
    h: int,
    exit_tiles: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Compute BFS path length from every walkable tile to the nearest tile in exit_tiles."""
    walkable = _make_walkable_fn(is_walkable)
    exit_set = {tuple(t) for t in exit_tiles}
    dist: dict[tuple[int, int], int] = {}
    q: deque[tuple[tuple[int, int], int]] = deque()

    for x, y in exit_set:
        if 0 <= x < w and 0 <= y < h:
            dist[(x, y)] = 0
            q.append(((x, y), 0))

    while q:
        (cx, cy), d = q.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in dist and walkable(nx, ny):
                dist[(nx, ny)] = d + 1
                q.append(((nx, ny), d + 1))

    return dist


def select_tiles(
    is_walkable: Callable[[int, int], bool] | Any,
    w: int,
    h: int,
    start: tuple[int, int],
    exit_tiles: Iterable[tuple[int, int]],
    excluded: Iterable[tuple[int, int]],
    n: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """Pure function selecting tiles stratified by distance to exit.

    - BFS (4-neighbour) over is_walkable from start to find reachable tiles.
    - dist is the BFS path length from each tile to the nearest tile in exit_tiles.
    - Exclude:
      - the exit tiles themselves;
      - every tile in excluded (doors / other exits);
      - tiles with no path to an exit.
    - Stratify by dist into 4 quartile bins and sample n // 4 tiles per bin without replacement.
    - Returns list of (x, y, dist) sorted by dist, then (x, y).
    """
    walkable = _make_walkable_fn(is_walkable)
    exit_set = {tuple(t) for t in exit_tiles}
    excluded_set = {tuple(t) for t in excluded}
    start_tuple = tuple(start)

    # 1. Reachable tiles from start
    reachable: set[tuple[int, int]] = set()
    if 0 <= start_tuple[0] < w and 0 <= start_tuple[1] < h and walkable(start_tuple[0], start_tuple[1]):
        q_reach: deque[tuple[int, int]] = deque([start_tuple])
        reachable.add(start_tuple)
        while q_reach:
            cx, cy = q_reach.popleft()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in reachable and walkable(nx, ny):
                    reachable.add((nx, ny))
                    q_reach.append((nx, ny))

    # 2. BFS distances to nearest exit
    dist = compute_exit_distances(walkable, w, h, exit_set)

    # 3. Filter candidates
    candidates = [
        t for t in reachable
        if t not in exit_set and t not in excluded_set and t in dist
    ]
    candidates.sort(key=lambda t: (dist[t], t[0], t[1]))

    k = n // 4
    if k == 0 or not candidates:
        return []

    dists = [dist[t] for t in candidates]
    q0, q1, q2, q3, q4 = np.quantile(dists, [0, 0.25, 0.5, 0.75, 1.0])

    b0 = [t for t in candidates if dist[t] <= q1]
    b1 = [t for t in candidates if q1 < dist[t] <= q2]
    b2 = [t for t in candidates if q2 < dist[t] <= q3]
    b3 = [t for t in candidates if q3 < dist[t] <= q4]
    bins = [b0, b1, b2, b3]

    # If tied quantile boundaries make any bin too small, fall back to rank-based index chunks
    if any(len(b) < k for b in bins):
        chunks = np.array_split(candidates, 4)
        bins = [list(c) for c in chunks]

    selected: list[tuple[int, int]] = []
    for b in bins:
        b_sorted = sorted(b, key=lambda t: (dist[t], t[0], t[1]))
        sample_size = min(k, len(b_sorted))
        if sample_size > 0:
            chosen_indices = rng.choice(len(b_sorted), size=sample_size, replace=False)
            for idx in chosen_indices:
                selected.append(b_sorted[idx])

    result = [(x, y, dist[(x, y)]) for (x, y) in selected]
    result.sort(key=lambda item: (item[2], item[0], item[1]))
    return result


def parse_extra_tiles(s: str) -> list[tuple[int, int]]:
    """Parse comma/semicolon/space-separated coordinate pairs, e.g. '23,26;10,12' or '23,26'."""
    if not s or s.strip().lower() == "none":
        return []
    parts = s.replace(";", " ").replace("|", " ").split()
    tiles = []
    for part in parts:
        if not part:
            continue
        sub = part.split(",")
        if len(sub) == 2:
            tiles.append((int(sub[0]), int(sub[1])))
    return tiles


def parse_policies(s: str) -> list[tuple[str, str, bool, str]]:
    """Parse comma-separated policy specs into (raw_spec, name, is_ppo, family)."""
    policies = []
    items = [p.strip() for p in s.split(",") if p.strip()]
    for item in items:
        if item.startswith("ppo_real:"):
            name = item[len("ppo_real:"):].strip()
            is_ppo = True
            family = "ppo_real"
        elif item.startswith("agent_rl:"):
            name = item[len("agent_rl:"):].strip()
            is_ppo = False
            family = "wm31"
        else:
            name = item
            # Auto-detect PPO if folder exists in work/ppo_real and not in work/agent_rl
            if (Path("work/ppo_real") / name).exists() and not (Path("work/agent_rl") / name).exists():
                is_ppo = True
                family = "ppo_real"
            else:
                is_ppo = False
                family = "wm31"
        policies.append((item, name, is_ppo, family))
    return policies


def get_mpc_stem(task: str, policy_name: str, is_ppo: bool, seed: int, state_path: Path | str | None) -> str:
    """Compute the exact stem matching lab/agent/mpc.py lines 198-199 for policy=learned, geodesic=False."""
    rl_dir = "work/ppo_real" if is_ppo else "work/agent_rl"
    prefix = "ppo_real_" if Path(rl_dir).name == "ppo_real" else ""
    tag = f"_{prefix}{policy_name}"
    state_suffix = f"_from_{Path(state_path).stem}" if state_path else ""
    return f"{task}_learned{tag}_s{seed}{state_suffix}"


def create_or_reuse_start_state(
    task_state_path: str,
    map_id: int,
    target: tuple[int, int],
    out_path: Path,
    budget: int = 400,
) -> tuple[bool, str]:
    """Navigate to target tile and snapshot start state; reuses if state file already exists."""
    if out_path.exists():
        return True, "reused"

    with PokemonRed(state=task_state_path) as env:
        r = navigate_to(env, map_id, target_tiles=[target], budget=budget)
        for _ in range(4):
            env.step("pass")
        _, ram = env.observe()
        actual = (int(ram["map_id"]), int(ram["x"]), int(ram["y"]))
        expected = (map_id, target[0], target[1])
        if actual != expected:
            reason = (
                f"RAM mismatch: ended at {actual} instead of {expected} "
                f"(success={r.get('success')}, presses={r.get('presses')})"
            )
            return False, reason

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(env.snapshot())
        return True, "saved"


def run_single_eval(cmd: list[str], log_path: Path) -> None:
    """Execute mpc subprocess with CPU environment and logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env_vars = os.environ.copy()
    env_vars["OMP_NUM_THREADS"] = "2"
    env_vars["CUDA_VISIBLE_DEVICES"] = ""

    with open(log_path, "w", encoding="utf-8") as fh:
        res = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env_vars,
            check=False,
        )

    if res.returncode != 0:
        raise RuntimeError(
            f"Evaluation subprocess failed with return code {res.returncode}. "
            f"Command: {' '.join(cmd)}\nSee log: {log_path}"
        )


def main():
    parser = argparse.ArgumentParser(description="Start-tile robustness sweep for leg policies.")
    parser.add_argument("--task", default="to_route2", help="Task name from lab.agent.tasks")
    parser.add_argument(
        "--policies",
        default="to_route2_wm31,to_route2_wm31_s1,to_route2_wm31_s2,ppo_real:to_route2_geo_s42,ppo_real:to_route2_geo_s1,ppo_real:to_route2_geo_s2",
        help="Comma-separated policy names or ppo_real:name",
    )
    parser.add_argument("--n-tiles", type=int, default=24, help="Number of tiles to sample (stratified into 4 quartile bins)")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per evaluation")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent subprocess workers")
    parser.add_argument("--extra-tiles", default="23,26", help="Extra start tiles to always include (comma/semicolon separated)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for tile selection")
    args = parser.parse_args()

    if args.task not in TASKS:
        raise ValueError(f"Unknown task: {args.task}. Available: {sorted(TASKS)}")

    task_spec = TASKS[args.task]
    task_state_path = task_spec["state"]
    extra_tiles = parse_extra_tiles(args.extra_tiles)
    policies = parse_policies(args.policies)

    # 1. Load task start state and build walkability grid
    with PokemonRed(state=task_state_path) as env:
        _, ram = env.observe()
        map_id = int(ram["map_id"])
        train_start = (int(ram["x"]), int(ram["y"]))
        w, h, walkable = build_walkability_grid(env)

    exit_tiles = task_spec["exits"].get(map_id, [])
    if not exit_tiles:
        raise ValueError(f"No exits defined for map {map_id} in task {args.task}")

    excluded = [
        t for m, tiles in task_spec["exits"].items()
        if m == map_id for t in tiles if t not in exit_tiles
    ]

    # Compute distances across the full map for reporting & extra tiles
    all_dist = compute_exit_distances(walkable, w, h, exit_tiles)

    # Pure tile selection
    rng = np.random.default_rng(args.seed)
    sampled = select_tiles(walkable, w, h, train_start, exit_tiles, excluded, args.n_tiles, rng)

    # Compute stats for full selection
    # All reachable tiles from train_start:
    q_reach: deque[tuple[int, int]] = deque([train_start])
    full_reachable: set[tuple[int, int]] = {train_start}
    while q_reach:
        cx, cy = q_reach.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in full_reachable and walkable(nx, ny):
                full_reachable.add((nx, ny))
                q_reach.append((nx, ny))

    full_candidates = [
        t for t in full_reachable
        if t not in set(exit_tiles) and t not in set(excluded) and t in all_dist
    ]
    full_dists = [all_dist[t] for t in full_candidates]
    q0, q1, q2, q3, q4 = np.quantile(full_dists, [0, 0.25, 0.5, 0.75, 1.0])

    print("=== Tile Selection Stats ===")
    print(f"Reachable tiles: {len(full_reachable)}")
    print(f"Dist range: [{min(full_dists)}, {max(full_dists)}]")
    print(f"Quartile edges: [{q0}, {q1}, {q2}, {q3}, {q4}]")
    print(f"Sampled {len(sampled)} tiles from {len(full_candidates)} candidates.")

    # Combine sampled tiles, task start tile, and extra tiles
    sweep_tiles_map: dict[tuple[int, int], dict[str, Any]] = {}
    for x, y, d in sampled:
        sweep_tiles_map[(x, y)] = {
            "x": x,
            "y": y,
            "dist_to_exit": d,
            "is_train_start": False,
            "is_extra": False,
        }

    # Add train_start
    d_train = all_dist.get(train_start)
    if d_train is not None:
        if train_start in sweep_tiles_map:
            sweep_tiles_map[train_start]["is_train_start"] = True
        else:
            sweep_tiles_map[train_start] = {
                "x": train_start[0],
                "y": train_start[1],
                "dist_to_exit": d_train,
                "is_train_start": True,
                "is_extra": False,
            }

    # Add extra_tiles
    for extra in extra_tiles:
        d_extra = all_dist.get(extra)
        if d_extra is not None:
            if extra in sweep_tiles_map:
                sweep_tiles_map[extra]["is_extra"] = True
            else:
                sweep_tiles_map[extra] = {
                    "x": extra[0],
                    "y": extra[1],
                    "dist_to_exit": d_extra,
                    "is_train_start": False,
                    "is_extra": True,
                }

    # Sort tiles by dist_to_exit, then (x, y)
    sweep_tiles = sorted(
        sweep_tiles_map.values(),
        key=lambda item: (item["dist_to_exit"], item["x"], item["y"]),
    )

    # 2. Start states generation
    state_dir = Path("work/route_states") / f"sweep_{args.task}"
    state_dir.mkdir(parents=True, exist_ok=True)
    skipped: list[dict[str, Any]] = []
    valid_tiles: list[dict[str, Any]] = []

    for tile_info in sweep_tiles:
        tile = (tile_info["x"], tile_info["y"])
        state_file = state_dir / f"map{map_id}_{tile[0]}_{tile[1]}.state"
        ok, reason = create_or_reuse_start_state(task_state_path, map_id, tile, state_file, budget=400)
        if ok:
            tile_info["state_path"] = state_file
            valid_tiles.append(tile_info)
        else:
            print(f"Skipping tile {tile}: {reason}")
            skipped.append({"tile": [tile[0], tile[1]], "reason": reason})

    skipped_file = state_dir / "skipped.json"
    skipped_file.write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    agent_dir = Path("work/agent") / f"sweep_{args.task}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    # 3. Subprocess evaluations
    jobs_to_run = []
    eval_specs = []

    for tile_info in valid_tiles:
        state_path = tile_info["state_path"]
        for raw_spec, policy_name, is_ppo, family in policies:
            stem = get_mpc_stem(args.task, policy_name, is_ppo, seed=200, state_path=state_path)
            json_path = Path("work/agent") / f"{stem}.json"
            log_path = agent_dir / "logs" / f"{stem}.log"

            eval_specs.append({
                "tile": tile_info,
                "raw_spec": raw_spec,
                "policy_name": policy_name,
                "is_ppo": is_ppo,
                "family": family,
                "stem": stem,
                "json_path": json_path,
            })

            # Check resume condition
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as fh:
                        existing = json.load(fh)
                    if existing.get("episodes") == args.episodes:
                        continue
                except Exception:
                    pass

            cmd = [
                sys.executable,
                "-m", "lab.agent.mpc",
                "--task", args.task,
                "--policy", "learned",
                "--rl-name", policy_name,
            ]
            if is_ppo:
                cmd.extend(["--rl-dir", "work/ppo_real"])
            cmd.extend([
                "--episodes", str(args.episodes),
                "--max-steps", "1000",
                "--seed", "200",
                "--device", "cpu",
                "--state", str(state_path),
            ])

            jobs_to_run.append({"cmd": cmd, "log_path": log_path, "stem": stem})

    print(f"Evaluations: {len(eval_specs)} total, {len(jobs_to_run)} to execute ({len(eval_specs) - len(jobs_to_run)} resumed).")

    if jobs_to_run:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_single_eval, job["cmd"], job["log_path"]): job
                for job in jobs_to_run
            }
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                future.result()

    # 4. Aggregation
    csv_rows = []
    matrix: dict[tuple[int, int], dict[str, str]] = {}

    for spec in eval_specs:
        tile = spec["tile"]
        json_path = spec["json_path"]
        if not json_path.exists():
            raise FileNotFoundError(f"Expected output file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        successes = int(data["successes"])
        episodes = int(data["episodes"])
        success_rate = float(data["success_rate"])
        median_steps = data.get("median_steps_when_successful")

        row = {
            "x": tile["x"],
            "y": tile["y"],
            "dist_to_exit": tile["dist_to_exit"],
            "is_train_start": tile["is_train_start"],
            "is_extra": tile["is_extra"],
            "policy": spec["policy_name"],
            "family": spec["family"],
            "successes": successes,
            "episodes": episodes,
            "success_rate": success_rate,
            "median_steps_when_successful": f"{median_steps:.1f}" if median_steps is not None else "",
        }
        csv_rows.append(row)

        coord = (tile["x"], tile["y"])
        if coord not in matrix:
            matrix[coord] = {}
        matrix[coord][spec["raw_spec"]] = f"{successes}/{episodes}"

    csv_rows.sort(key=lambda r: (r["dist_to_exit"], r["x"], r["y"], r["policy"]))

    csv_path = agent_dir / "summary.csv"
    fieldnames = [
        "x",
        "y",
        "dist_to_exit",
        "is_train_start",
        "is_extra",
        "policy",
        "family",
        "successes",
        "episodes",
        "success_rate",
        "median_steps_when_successful",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    # Write table.md
    policy_headers = [p[0] for p in policies]
    table_lines = [
        "| tile | dist | " + " | ".join(policy_headers) + " |",
        "|---|---| " + " | ".join(["---"] * len(policy_headers)) + " |",
    ]

    for tile in valid_tiles:
        coord = (tile["x"], tile["y"])
        cells = [matrix.get(coord, {}).get(p, "n/a") for p in policy_headers]
        row_str = f"| ({coord[0]}, {coord[1]}) | {tile['dist_to_exit']} | " + " | ".join(cells) + " |"
        table_lines.append(row_str)

    table_md = "\n".join(table_lines) + "\n"
    (agent_dir / "table.md").write_text(table_md, encoding="utf-8")

    # Print table to stdout
    print("\n" + table_md)


if __name__ == "__main__":
    main()
