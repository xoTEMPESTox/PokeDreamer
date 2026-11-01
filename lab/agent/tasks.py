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


def exit_distance(maps, xs, ys, exits):
    """Manhattan distance to the nearest exit tile of each element's map (arrays of equal shape); maps without
    exits get distance 0 and are handled by the caller."""
    import numpy as np
    maps, xs, ys = np.asarray(maps), np.asarray(xs), np.asarray(ys)
    out = np.zeros(maps.shape, dtype=float)
    for m, tiles in exits.items():
        sel = maps == m
        if sel.any():
            out[sel] = np.min([np.abs(xs[sel] - x) + np.abs(ys[sel] - y) for x, y in tiles], axis=0)
    return out
