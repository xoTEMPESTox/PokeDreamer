"""Navigation tasks for the real-game agents. Exit tiles come from the recorded data's own map transitions
(e.g. Pallet Town -> Route 1 happened from (10,0) / (11,0) 88 times); several exits per map are allowed.

`exits` also covers buildings an agent can wander into on the way ("detours": Oak's Lab door (4,11)/(5,11),
70 recorded exits; Blue's house (3,7), 39; Red's house stairs and door), so every policy knows the way back out.
`train_maps` are the maps whose real contexts start imagined rollouts (balanced per map); default: all exit maps.

leave_house: bedroom -> 1st floor -> Pallet Town (no Pokemon needed).
route1:      Pallet Town -> Route 1, from the scripted starter-Pokemon state (Oak no longer stops you).
viridian:    Pallet Town -> Route 1 -> Viridian City, through tall grass (wild battles) and ledges.
to_route2:   Viridian City -> Route 2 south, through Viridian City (detours: Pokecenter, Mart, School, House, Route 22).
forest:      Viridian Forest South Gate -> Viridian Forest -> North Gate (detours: Route 2 south).
to_pewter:   Viridian Forest North Gate -> Route 2 north -> Pewter City (detours: Viridian Forest, South Gate).
to_gym:      Pewter City entry -> Pewter Gym (detours: Pokecenter, Mart, Houses, Museum, Route 2).
"""

_PALLET_DETOURS = {40: [(4, 11), (5, 11)], 39: [(3, 7)], 38: [(7, 1)], 37: [(2, 7)]}
_VIRIDIAN_DETOURS = {
    41: [(3, 7), (4, 7)],                       # Pokecenter (90 transitions: (3,7) 51, (4,7) 39)
    42: [(3, 7), (4, 7)],                       # Mart (77 transitions: (4,7) 42, (3,7) 35)
    43: [(2, 7), (3, 7)],                       # School (59 transitions: (2,7) 41, (3,7) 18)
    44: [(2, 7), (3, 7)],                       # House (87 transitions: (2,7) 63, (3,7) 24)
    33: [(39, 6), (39, 7), (39, 8), (39, 9)],   # Route 22 (59 transitions: (39,9) 25, (39,6) 21, (39,8) 6, (39,7) 6)
    12: [(10, 0), (11, 0)],                     # Route 1 (118 transitions: (10,0) 70, (11,0) 44)
    **_PALLET_DETOURS,
}
_PEWTER_DETOURS = {
    58: [(3, 7), (4, 7)],                       # Pokecenter (79 transitions: (3,7) 49, (4,7) 30)
    55: [(2, 7), (3, 7)],                       # Mart (81 transitions: (2,7) 63, (3,7) 18)
    57: [(2, 7), (3, 7)],                       # House 1 (77 transitions: (2,7) 46, (3,7) 31)
    52: [(10, 7), (11, 7)],                     # Museum (73 transitions: (10,7) 53, (11,7) 20)
    56: [(3, 7), (4, 7)],                       # House 2 (70 transitions: (3,7) 38, (4,7) 32)
    13: [(8, 0), (9, 0)],                       # Route 2 north exit to Pewter (97 transitions)
}

TASKS = {
    "leave_house": {"state": "saves/bedroom.state", "exits": {38: [(7, 1)], 37: [(2, 7)]}, "success_map": 0},
    "route1": {"state": "saves/starter_pallet.state", "exits": {0: [(10, 0), (11, 0)], **_PALLET_DETOURS},
               "success_map": 12, "train_maps": [0, 40]},
    "viridian": {"state": "saves/starter_pallet.state",
                 "exits": {0: [(10, 0), (11, 0)], 12: [(10, 0), (11, 0)], **_PALLET_DETOURS},
                 "success_map": 1, "train_maps": [0, 12, 40]},
    # to_route2: Viridian City (1) -> Route 2 south (13). North exit (17,0) / (18,0) / (19,0) (278 transitions).
    "to_route2": {
        "state": "work/route_states/seed100_map1.state",
        "exits": {1: [(17, 0), (18, 0), (19, 0)], **_VIRIDIAN_DETOURS},
        "success_map": 13,
        "train_maps": [1],
    },
    # forest: South Gate (50) -> Viridian Forest (51) -> North Gate (47).
    # 50 -> 51 at (5,1) (52 transitions); 51 -> 47 at (1,0) / (2,0) (109 transitions: (1,0) 91, (2,0) 18).
    "forest": {
        "state": "work/route_states/seed100_map50.state",
        "exits": {50: [(5, 1)], 51: [(1, 0), (2, 0)], 13: [(3, 44)], 1: [(17, 0), (18, 0), (19, 0)], **_VIRIDIAN_DETOURS},
        "success_map": 47,
        "train_maps": [50, 51],
    },
    # to_pewter: North Gate (47) -> Route 2 north (13) -> Pewter City (2).
    # 47 -> 13 at (5,1) (68 transitions); 13 -> 2 at (8,0) / (9,0) (97 transitions: (9,0) 63, (8,0) 34).
    "to_pewter": {
        "state": "work/route_states/seed100_map47.state",
        "exits": {47: [(5, 1)], 13: [(8, 0), (9, 0)], 51: [(1, 0), (2, 0)], 50: [(5, 1)]},
        "success_map": 2,
        "train_maps": [47, 13],
    },
    # to_gym: Pewter City (2) -> Pewter Gym (54). Gym door at (16,18) (52 transitions).
    "to_gym": {
        "state": "work/route_states/seed100_map2.state",
        "exits": {2: [(16, 18)], **_PEWTER_DETOURS},
        "success_map": 54,
        "train_maps": [2],
    },
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


def train_geodesic_fields(exits, extended=False):
    """geodesic_fields from the train split of the 72x80 cache (work/cache_72x80.npz, work/splits.json),
    or the extended dataset (work/cache_ext_72x80.npz, work/splits_ext.json) when extended=True."""
    import numpy as np
    from lab.data import load
    data, splits = load(extended=extended)
    train = np.isin(data["episode_id"], splits["train"])
    return geodesic_fields(*(data[k][train] for k in ("map_ids", "xs", "ys", "episode_id")), exits)
