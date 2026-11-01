"""Path distance to an exit must follow the moves seen in the data (detours, one-way ledges), not a straight line."""
import numpy as np

from lab.agent.mpc import greedy_action
from lab.agent.tasks import exit_distance, geodesic_fields

# one episode on map 5 walking around a wall at (2, 1) to the exit (2, 0)
MAPS = np.full(5, 5)
XS = np.array([2, 1, 1, 1, 2])
YS = np.array([2, 2, 1, 0, 0])
EXITS = {5: [(2, 0)]}


def test_detour_and_unvisited_tiles():
    f = geodesic_fields(MAPS, XS, YS, np.zeros(5, int), EXITS, size=8)[5]
    assert f[2, 2] == 4 and f[1, 1] == 2 and f[0, 2] == 0     # [y, x]: around the wall, not 2 straight up
    assert f[1, 2] == 5                                       # the wall (never visited): worse than any known tile
    assert list(exit_distance([5, 5], [2, 1], [2, 1], EXITS, {5: f})) == [4, 2]
    assert list(exit_distance([5], [2], [2], EXITS)) == [2]   # Manhattan without fields


def test_one_way_ledge():
    # exit (1, 0); jump down the ledge (1, 0) -> (1, 1), then walk around: (2, 1), (2, 0), back to (1, 0)
    xs, ys = np.array([1, 1, 2, 2, 1]), np.array([0, 1, 1, 0, 0])
    f = geodesic_fields(np.full(5, 7), xs, ys, np.zeros(5, int), {7: [(1, 0)]}, size=4)[7]
    assert f[1, 1] == 3                                       # not 1: the ledge can't be climbed


def test_moves_across_episodes_are_not_edges():
    eps = np.array([0, 1, 1, 1, 1])                           # the (2,2)->(1,2) move straddles two episodes
    f = geodesic_fields(MAPS, XS, YS, eps, EXITS, size=8)[5]
    assert f[2, 1] == 3 and f[2, 2] == f[7, 7]                # (2,2) is off the observed graph


def test_greedy_follows_the_field():
    f = geodesic_fields(MAPS, XS, YS, np.zeros(5, int), EXITS, size=8)[5]
    ram = {"map_id": 5, "x": 2, "y": 2}
    a = greedy_action(ram, None, 0, np.random.default_rng(0), EXITS, {5: f})
    assert a == 1                                             # left (around the wall), not up into it
