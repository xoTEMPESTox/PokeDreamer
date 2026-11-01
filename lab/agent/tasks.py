"""Navigation tasks for the real-game agents. Exit tiles come from the recorded data's own map transitions
(e.g. Pallet Town -> Route 1 happened from (10,0) / (11,0) 88 times); several exits per map are allowed.

`exits` also covers buildings an agent can wander into on the way ("detours": Oak's Lab door (4,11)/(5,11),
70 recorded exits; Blue's house (3,7), 39; Red's house stairs and door), so every policy knows the way back out.
`train_maps` are the maps whose real contexts start imagined rollouts (balanced per map); default: all exit maps.

leave_house: bedroom -> 1st floor -> Pallet Town (no Pokemon needed).
route1:      Pallet Town -> Route 1, from the scripted starter-Pokemon state (Oak no longer stops you).
viridian:    Pallet Town -> Route 1 -> Viridian City, through tall grass (wild battles) and ledges.
"""

_PALLET_DETOURS = {40: [(4, 11), (5, 11)], 39: [(3, 7)], 38: [(7, 1)], 37: [(2, 7)]}

TASKS = {
    "leave_house": {"state": "saves/bedroom.state", "exits": {38: [(7, 1)], 37: [(2, 7)]}, "success_map": 0},
    "route1": {"state": "saves/starter_pallet.state", "exits": {0: [(10, 0), (11, 0)], **_PALLET_DETOURS},
               "success_map": 12, "train_maps": [0, 40]},
    "viridian": {"state": "saves/starter_pallet.state",
                 "exits": {0: [(10, 0), (11, 0)], 12: [(10, 0), (11, 0)], **_PALLET_DETOURS},
                 "success_map": 1, "train_maps": [0, 12, 40]},
}


def exit_distance(maps, xs, ys, exits, fields=None):
    """Distance to the nearest exit tile of each element's map (arrays of equal shape): from `fields`
    (geodesic_fields) where given, else Manhattan; maps without exits get distance 0 and are handled by the caller."""
    import numpy as np
    maps, xs, ys = np.asarray(maps), np.asarray(xs), np.asarray(ys)
    out = np.zeros(maps.shape, dtype=float)
    for m, tiles in exits.items():
        sel = maps == m
        if sel.any() and fields is not None and m in fields:
            f = fields[m]
            out[sel] = f[np.clip(ys[sel], 0, f.shape[0] - 1), np.clip(xs[sel], 0, f.shape[1] - 1)]
        elif sel.any():
            out[sel] = np.min([np.abs(xs[sel] - x) + np.abs(ys[sel] - y) for x, y in tiles], axis=0)
    return out


def geodesic_fields(maps, xs, ys, episodes, exits, size=64):
    """Per map in `exits`: (size, size) array [y, x] of the fewest one-tile moves to its nearest exit tile, over
    the moves observed in the data (directed, so a ledge only counts downhill). Tiles with no observed path get
    the map's largest distance + 1, so walls and hallucinated positions never look closer than the real route.
    Needed for Route 1: from row 28 the only way north is at x=7, and with Manhattan distance the viridian policy
    stalled at x 10-15 (0 of the first 72 real episodes)."""
    from collections import deque
    import numpy as np
    maps, xs, ys, episodes = (np.asarray(a).astype(np.int64) for a in (maps, xs, ys, episodes))
    move = ((episodes[:-1] == episodes[1:]) & (maps[:-1] == maps[1:])
            & (np.abs(np.diff(xs)) + np.abs(np.diff(ys)) == 1))
    fields = {}
    for m, tiles in exits.items():
        into = {}
        for i in np.flatnonzero(move & (maps[:-1] == m)):
            into.setdefault((xs[i + 1], ys[i + 1]), set()).add((xs[i], ys[i]))
        dist = {tuple(t): 0 for t in tiles}
        queue = deque(dist)
        while queue:
            t = queue.popleft()
            for s in into.get(t, ()):
                if s not in dist:
                    dist[s] = dist[t] + 1
                    queue.append(s)
        f = np.full((size, size), max(dist.values()) + 1)
        for (x, y), d in dist.items():
            if 0 <= x < size and 0 <= y < size:
                f[y, x] = d
        fields[m] = f
    return fields


def train_geodesic_fields(exits):
    """geodesic_fields from the train split of the 72x80 cache (work/cache_72x80.npz, work/splits.json)."""
    import json
    import numpy as np
    from lab.data import CACHE
    d = np.load(CACHE)
    train = np.isin(d["episode_id"], json.load(open("work/splits.json"))["train"])
    return geodesic_fields(*(d[k][train] for k in ("map_ids", "xs", "ys", "episode_id")), exits)
