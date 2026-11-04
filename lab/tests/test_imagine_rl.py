"""Reward and return logic of the imagination-trained policy (lab/agent/imagine_rl.py)."""
import numpy as np
import torch

from lab.agent.imagine_rl import HALLUCINATION_PENALTY, STEP_COST, WARP_BONUS, lambda_returns, rewards_from_probe


def test_progress_hallucinated_warp_and_real_warp():
    # bedroom (38), exit (7, 1); start at (3, 6): distance 9
    maps = np.array([[38, 38, 38], [38, 37, 37], [38, 37, 37]])
    xs = np.array([[4, 5, 5], [3, 7, 7], [7, 7, 7]])
    ys = np.array([[6, 6, 5], [6, 1, 1], [1, 1, 1]])
    start = np.array([38, 38, 38])
    r, alive, cont = rewards_from_probe(maps, xs, ys, start, np.array([3, 3, 6]), np.array([6, 6, 1]))
    np.testing.assert_allclose(r[0], [1 - STEP_COST] * 3)                 # one tile closer each step
    assert r[1, 1] == -HALLUCINATION_PENALTY and alive[1, 2] == 0 and cont[1, 1] == 0   # teleport mid-room
    assert r[2, 1] == WARP_BONUS                                          # map change right after the exit tile


def test_lambda_returns_stop_at_termination():
    r = torch.tensor([[1.0, 1.0, 100.0]])
    v = torch.zeros(1, 3)
    cont = torch.tensor([[1.0, 0.0, 0.0]])                                 # rollout ends at step 1
    g = lambda_returns(r, v, cont, bootstrap=torch.tensor([50.0]))
    assert torch.allclose(g[0, 1], torch.tensor(1.0))                     # no reward or value after the end


def test_battle_ends_rollout_without_penalty_and_multi_exit_tasks():
    from lab.agent.tasks import TASKS
    exits = TASKS["route1"]["exits"]                      # Pallet Town -> Route 1 at (10, 0) or (11, 0)
    maps = np.array([[0, 0, 0], [0, 0, 0]])
    xs = np.array([[11, 11, 11], [10, 10, 10]])
    ys = np.array([[3, 2, 1], [5, 5, 5]])
    battle = np.array([[False, False, False], [False, True, True]])
    r, alive, cont = rewards_from_probe(maps, xs, ys, np.array([0, 0]), np.array([11, 10]), np.array([4, 5]), exits, battle)
    np.testing.assert_allclose(r[0], [1 - STEP_COST] * 3)                     # nearest exit (11, 0): one tile closer per step
    assert r[1, 1] == -STEP_COST and alive[1, 2] == 0 and cont[1, 1] == 0      # battle: ends, no warp bonus or penalty


def test_step_clip_bounds_misread_jumps():
    # bedroom exit (7, 1); the probe misreads step 2 as 6 tiles away, then reads the true tile again
    maps = np.full((1, 3), 38)
    xs, ys = np.array([[4, 4, 4]]), np.array([[6, 9, 6]])          # true walk would be (4,6) -> (4,6) -> (4,6)
    start = (np.array([38]), np.array([4]), np.array([6]))
    r, _, _ = rewards_from_probe(maps, xs, ys, *start, step_clip=1.0)
    np.testing.assert_allclose(r[0], [-STEP_COST, -1 - STEP_COST, 1 - STEP_COST])   # +-1 at most per press
    r0, _, _ = rewards_from_probe(maps, xs, ys, *start)
    np.testing.assert_allclose(r0[0], [-STEP_COST, -3 - STEP_COST, 3 - STEP_COST])  # unclipped default unchanged


def _ledge_field():
    # Viridian (map 1) around the to_route2 trap, values from the real ext-train geodesic field [y, x]:
    # ledge row y=27 (one-way down, (18,27) is only seen mid-jump), walkable gap at x=19; unobserved tiles = max (50)
    f = np.full((80, 80), 50.0)
    for (x, y), d in {(18, 25): 25, (18, 26): 26, (19, 26): 27, (18, 27): 31, (19, 27): 28, (18, 28): 30,
                      (19, 28): 29}.items():
        f[y, x] = d
    return {1: [(17, 0), (18, 0), (19, 0)]}, {1: f}


def test_ledge_loop_pays_under_step_clip_but_not_under_move_gate():
    # the real loop of the to_route2 seed-0/1 trap: jump the ledge at (18,26), walk back up through the gap
    exits, fields = _ledge_field()
    loop = [(18, 27), (18, 28), (19, 28), (19, 27), (19, 26), (18, 26)]
    maps, xs, ys = np.ones((1, 6), int), np.array([[x for x, _ in loop]]), np.array([[y for _, y in loop]])
    start = (np.array([1]), np.array([18]), np.array([26]))
    r_clip, _, _ = rewards_from_probe(maps, xs, ys, *start, exits, None, fields, step_clip=1.0)
    assert (r_clip + STEP_COST).sum() > 3.5                      # the bug: a closed loop earns progress
    r_gate, _, _ = rewards_from_probe(maps, xs, ys, *start, exits, None, fields, move_gate=2)
    np.testing.assert_allclose((r_gate + STEP_COST).sum(), 0.0)   # potential-based: a closed loop earns nothing
    np.testing.assert_allclose(r_gate[0, 0] + STEP_COST, -5.0)    # the jump costs its real distance


def test_move_gate_rejects_misreads_and_keeps_its_reference():
    exits, fields = _ledge_field()
    start = (np.array([1]), np.array([19]), np.array([26]))
    # (19,26) -> misread far away (19,2) -> misread into an unobserved tile (20,26) -> true (18,26) -> (18,25)
    xs, ys = np.array([[19, 20, 18, 18]]), np.array([[2, 26, 26, 25]])
    r, _, _ = rewards_from_probe(np.ones((1, 4), int), xs, ys, *start, exits, None, fields, move_gate=2)
    np.testing.assert_allclose(r[0] + STEP_COST, [0.0, 0.0, 1.0, 1.0])   # misreads: no progress; then 27->26->25
