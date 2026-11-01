"""Counts map transitions (m -> m') and the last tile on m before each one in the collected data.

Used to find the exit and detour tiles for navigation tasks.
At warps, Game Boy RAM updates wCurMap 1-2 steps before player x/y coordinates in RAM,
so the last tile on m is read 1-2 steps before the destination map's coordinates appear.
"""
from collections import Counter, defaultdict
import numpy as np


def count_transitions(maps_or_data, xs=None, ys=None, episodes=None, episode_starts=None,
                      skip_first=5, step_back=1):
    """Count transitions (m -> m') and the pre-transition tile on m.

    Args:
        maps_or_data: dict from lab.data.load() or array of map IDs.
        xs: array of X coordinates (if maps_or_data is an array).
        ys: array of Y coordinates (if maps_or_data is an array).
        episodes: array of episode IDs.
        episode_starts: boolean array marking episode start steps.
        skip_first: number of initial steps in each episode to ignore (clears state load artifacts).
        step_back: how many steps before the map change to read the tile on m (default 1).

    Returns:
        dict mapping (m_from, m_to) -> Counter of (x, y) exit tiles.
    """
    if isinstance(maps_or_data, dict):
        maps = np.asarray(maps_or_data["map_ids"])
        xs = np.asarray(maps_or_data["xs"])
        ys = np.asarray(maps_or_data["ys"])
        episodes = np.asarray(maps_or_data.get("episode_id", np.zeros(len(maps), int)))
        episode_starts = maps_or_data.get("episode_starts", None)
    else:
        maps = np.asarray(maps_or_data)
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        episodes = np.zeros(len(maps), int) if episodes is None else np.asarray(episodes)

    if episode_starts is None:
        episode_starts = np.zeros(len(maps), bool)
        episode_starts[0] = True
        episode_starts[1:] = episodes[1:] != episodes[:-1]
    else:
        episode_starts = np.asarray(episode_starts, bool)

    step_in_ep = np.zeros(len(maps), int)
    cur = 0
    for k in range(len(maps)):
        cur = 0 if episode_starts[k] else cur + 1
        step_in_ep[k] = cur

    change = (episodes[1:] == episodes[:-1]) & (maps[1:] != maps[:-1])
    idx = np.where(change)[0] + 1

    trans_counts = defaultdict(Counter)
    for i in idx:
        if step_in_ep[i] >= skip_first:
            target = i - step_back
            if target >= 0 and episodes[target] == episodes[i]:
                m1, m2 = int(maps[i - 1]), int(maps[i])
                tile = (int(xs[target]), int(ys[target]))
                trans_counts[(m1, m2)][tile] += 1

    return trans_counts


def main():
    from lab.data import load
    data, _ = load(extended=True)
    counts = count_transitions(data, skip_first=5, step_back=1)

    print("# Map Transitions in Extended Dataset")
    print()
    print("| Transition | Total | Exit tiles (tile: count) |")
    print("|---|---|---|")
    route_pairs = [
        (1, 13), (41, 1), (42, 1), (43, 1), (44, 1), (33, 1), (12, 1),
        (50, 51), (51, 47), (13, 50), (50, 13), (51, 50),
        (47, 13), (13, 2), (47, 51), (13, 47),
        (2, 54), (58, 2), (55, 2), (57, 2), (52, 2), (56, 2), (54, 2), (2, 13)
    ]
    for p in route_pairs:
        cnts = counts[p]
        tot = sum(cnts.values())
        tiles_str = ", ".join(f"{t}: {c}" for t, c in cnts.most_common(5))
        print(f"| {p[0]} -> {p[1]} | {tot} | {tiles_str} |")


if __name__ == "__main__":
    main()
