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
