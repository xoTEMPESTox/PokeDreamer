"""lab.video.dream: world-model frames aligned step by step with a recorded trajectory."""
import numpy as np
import torch

from lab.dwm.model import CTX
from lab.video.dream import dream_frames


class EchoModel:
    """Stand-in world model: each imagined frame is the last context frame plus the action (in [-1, 1] units)."""

    def rollout(self, ctx, actions, steps=3):
        k_total = actions.shape[1] - CTX + 1
        last = ctx[:, -1:]
        return torch.cat([last + actions[:, CTX - 1 + k:CTX + k, None, None].float() / 127.5 for k in range(k_total)], 1)


def test_chunks_are_anchored_on_real_frames_and_aligned():
    frames = np.zeros((40, 72, 80), np.uint8)
    frames[:, 0, 0] = np.arange(40) * 3              # mark each real frame
    actions = np.ones(40, np.int64)
    out, anchor = dream_frames(EchoModel(), frames, actions, start=8, end=24, k=8, device="cpu")
    assert out.shape == (16, 72, 80) and anchor.shape == (16,)
    assert anchor.tolist() == [True] + [False] * 7 + [True] + [False] * 7
    # step 8 is imagined from real frames 4..7: last real frame 7 (value 21) plus one action step (+1)
    assert int(out[0, 0, 0]) == 22 and int(out[8, 0, 0]) == 3 * 15 + 1   # chunk 2 re-anchors on real frame 15
    out_b, _ = dream_frames(EchoModel(), frames, actions, start=8, end=24, k=8, device="cpu", batch=1)
    assert np.array_equal(out, out_b)                                     # batching doesn't change the frames
