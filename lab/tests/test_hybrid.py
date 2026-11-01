"""Unit tests for the hybrid world model (RamDynamics and HybridDenoiser)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from lab.dwm.hybrid import HybridDenoiser, RamDynamics, rollout
from lab.dwm.model import CTX, Denoiser


def test_ram_dynamics_output_shapes():
    """Verify RamDynamics output shapes for forward, predict, and loss."""
    B = 3
    ram_dyn = RamDynamics(embed_dim=32, hidden_dim=64)

    ram_ctx = {
        "map": torch.randint(0, 256, (B, CTX)),
        "x": torch.randint(0, 64, (B, CTX)),
        "y": torch.randint(0, 80, (B, CTX)),
        "facing": torch.randint(0, 4, (B, CTX)),
        "battle": torch.randint(0, 2, (B, CTX)),
    }
    actions = torch.randint(0, 8, (B, CTX))

    logits = ram_dyn(ram_ctx, actions)
    assert logits["map"].shape == (B, 256)
    assert logits["x"].shape == (B, 64)
    assert logits["y"].shape == (B, 80)
    assert logits["facing"].shape == (B, 4)
    assert logits["battle"].shape == (B, 2)

    # Test predict argmax
    pred_argmax = ram_dyn.predict(ram_ctx, actions, sample=False)
    for k in ("map", "x", "y", "facing", "battle"):
        assert pred_argmax[k].shape == (B,)
        assert pred_argmax[k].dtype == torch.long

    # Test predict sample
    pred_sample = ram_dyn.predict(ram_ctx, actions, sample=True)
    for k in ("map", "x", "y", "facing", "battle"):
        assert pred_sample[k].shape == (B,)
        assert pred_sample[k].dtype == torch.long

    # Test loss computation
    target_ram = {
        "map": torch.randint(0, 256, (B,)),
        "x": torch.randint(0, 64, (B,)),
        "y": torch.randint(0, 80, (B,)),
        "facing": torch.randint(0, 4, (B,)),
        "battle": torch.randint(0, 2, (B,)),
    }
    loss = ram_dyn.loss(ram_ctx, actions, target_ram)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    # Test with aliases and raw facing values (0, 4, 8, 12)
    ram_alias = {
        "map_ids": np.random.randint(0, 256, (B, CTX), dtype=np.uint8),
        "xs": np.random.randint(0, 36, (B, CTX), dtype=np.uint8),
        "ys": np.random.randint(0, 72, (B, CTX), dtype=np.uint8),
        "facings": np.random.choice([0, 4, 8, 12], size=(B, CTX)).astype(np.uint8),
        "in_battles": np.random.choice([True, False], size=(B, CTX)),
    }
    logits_alias = ram_dyn(ram_alias, actions.numpy())
    assert logits_alias["map"].shape == (B, 256)


def test_hybrid_denoiser_loss_runs_and_finite():
    """Verify HybridDenoiser loss runs and is finite."""
    B = 2
    model = HybridDenoiser()
    frames = torch.randn(B, CTX + 1, 72, 80).clamp(-1, 1)
    actions = torch.randint(0, 8, (B, CTX))
    target_ram = {
        "map": torch.randint(0, 256, (B,)),
        "x": torch.randint(0, 64, (B,)),
        "y": torch.randint(0, 80, (B,)),
        "battle": torch.randint(0, 2, (B,)),
    }

    loss = model.loss(frames, actions, target_ram, ctx_aug=False)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # Test backward pass
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN gradient in {name}"


def test_rollout_shapes_and_stub_ram():
    """Verify rollout output shapes and that rollout uses predicted (not true) RAM after context."""
    B, K = 2, 5
    model = HybridDenoiser()

    class StubRam(nn.Module):
        def __init__(self):
            super().__init__()
            self.history = []

        def predict(self, ram_ctx, actions, sample=False):
            b = actions.shape[0]
            self.history.append({k: v.clone() for k, v in ram_ctx.items()})
            return {
                "map": torch.full((b,), 99, dtype=torch.long, device=actions.device),
                "x": torch.full((b,), 10, dtype=torch.long, device=actions.device),
                "y": torch.full((b,), 20, dtype=torch.long, device=actions.device),
                "facing": torch.zeros(b, dtype=torch.long, device=actions.device),
                "battle": torch.zeros(b, dtype=torch.long, device=actions.device),
            }

    stub = StubRam()
    frames_ctx = torch.zeros(B, CTX, 72, 80)
    ram_ctx = {
        "map": torch.zeros(B, CTX, dtype=torch.long),
        "x": torch.zeros(B, CTX, dtype=torch.long),
        "y": torch.zeros(B, CTX, dtype=torch.long),
        "facing": torch.zeros(B, CTX, dtype=torch.long),
        "battle": torch.zeros(B, CTX, dtype=torch.long),
    }
    actions = torch.zeros(B, CTX + K - 1, dtype=torch.long)

    frames, pred_ram = model.rollout(frames_ctx, ram_ctx, actions, steps=1, ram_model=stub)

    # Check return shapes and types
    assert frames.shape == (B, K, 72, 80)
    assert isinstance(pred_ram, dict)
    assert pred_ram["map"].shape == (B, K)
    assert pred_ram["x"].shape == (B, K)
    assert pred_ram["y"].shape == (B, K)

    # Stub always predicts map 99 -> check that pred_ram is all 99
    assert (pred_ram["map"] == 99).all()
    assert (pred_ram["x"] == 10).all()
    assert (pred_ram["y"] == 20).all()

    # Check that after context, the RAM model received its own predictions as context
    assert len(stub.history) == K
    # At step 0, context was all 0
    assert (stub.history[0]["map"] == 0).all()
    # At step 1, the latest entry in context should be 99
    assert (stub.history[1]["map"][:, -1] == 99).all()
    # At step 4 (after 4 predicted steps), all CTX entries in context should be 99
    assert (stub.history[4]["map"] == 99).all()

    # Check top-level rollout wrapper also works
    f2, r2 = rollout(frames_ctx, ram_ctx, actions, steps=1, model=model, ram_model=stub)
    assert f2.shape == (B, K, 72, 80)
    assert (r2["map"] == 99).all()


def test_action_alignment_matches_denoiser():
    """Verify action alignment in HybridDenoiser.rollout exactly matches Denoiser.rollout."""
    B, K = 2, 6
    actions = torch.arange(CTX + K - 1, dtype=torch.long).unsqueeze(0).expand(B, -1)
    frames_ctx = torch.zeros(B, CTX, 72, 80)
    ram_ctx = {
        "map": torch.zeros(B, CTX, dtype=torch.long),
        "x": torch.zeros(B, CTX, dtype=torch.long),
        "y": torch.zeros(B, CTX, dtype=torch.long),
        "facing": torch.zeros(B, CTX, dtype=torch.long),
        "battle": torch.zeros(B, CTX, dtype=torch.long),
    }

    class ActionRecordDenoiser(Denoiser):
        def __init__(self):
            super().__init__()
            self.action_slices = []

        def sample(self, ctx, act, steps=3, aug_bucket=0):
            self.action_slices.append(act.clone())
            return torch.zeros(act.shape[0], 1, 72, 80)

    class ActionRecordHybrid(HybridDenoiser):
        def __init__(self):
            super().__init__()
            self.action_slices = []

        def sample(self, ctx, act, ram_target, steps=3, aug_bucket=0):
            self.action_slices.append(act.clone())
            return torch.zeros(act.shape[0], 1, 72, 80)

    class DummyRam(nn.Module):
        def predict(self, r_ctx, act, sample=False):
            b = act.shape[0]
            return {
                "map": torch.zeros(b, dtype=torch.long),
                "x": torch.zeros(b, dtype=torch.long),
                "y": torch.zeros(b, dtype=torch.long),
                "facing": torch.zeros(b, dtype=torch.long),
                "battle": torch.zeros(b, dtype=torch.long),
            }

    d = ActionRecordDenoiser()
    h = ActionRecordHybrid()
    r = DummyRam()

    _ = d.rollout(frames_ctx, actions, steps=1)
    _ = h.rollout(frames_ctx, ram_ctx, actions, steps=1, ram_model=r)

    assert len(d.action_slices) == K
    assert len(h.action_slices) == K

    for k in range(K):
        expected_act = actions[:, k : k + CTX]
        assert (d.action_slices[k] == expected_act).all()
        assert (h.action_slices[k] == expected_act).all()
        assert (d.action_slices[k] == h.action_slices[k]).all()
