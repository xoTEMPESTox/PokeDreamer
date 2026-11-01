"""The planner's scoring must not reward warps the world model hallucinates away from the exit tile."""
import numpy as np

from lab.agent.mpc import score


def test_progress_beats_standing_still_and_hallucinated_teleports():
    # current real state: bedroom (map 38) at (3, 6); exit (stairs) at (7, 1), distance 9
    maps = np.array([[38, 38, 38, 38],     # walks toward the stairs
                     [38, 38, 38, 38],     # stands still
                     [38, 37, 37, 37],     # "teleports" downstairs from mid-room
                     [37, 37, 37, 37]])    # "teleports" at the first step
    xs = np.array([[4, 5, 5, 5], [3, 3, 3, 3], [3, 7, 7, 7], [7, 7, 7, 7]])
    ys = np.array([[6, 6, 5, 4], [6, 6, 6, 6], [6, 1, 1, 1], [1, 1, 1, 1]])
    s = score(maps, xs, ys, 38, (3, 6))
    assert s[0] > s[1]                      # progress is rewarded
    assert s[2] <= s[1] and s[3] < s[1]     # mid-room teleports earn nothing


def test_map_change_next_to_the_exit_is_credited():
    # at (6, 1), one tile from the stairs; stepping right warps to the 1st floor in the real game
    maps = np.array([[38, 37, 37, 37], [38, 38, 38, 38]])
    xs = np.array([[7, 7, 7, 7], [5, 4, 3, 2]])
    ys = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    s = score(maps, xs, ys, 38, (6, 1))
    assert s[0] == 10.0 and s[0] > s[1]
