"""lab.export_weights: published checkpoints keep only what the loaders read."""
import torch

from lab.export_weights import strip_checkpoint


def test_world_model_keeps_ema_and_counters():
    ck = {"model": {"w": torch.ones(2)}, "ema": {"w": torch.zeros(2)}, "opt": {"state": {}}, "step": 5, "elapsed": 1.0}
    out = strip_checkpoint(ck)
    assert set(out) == {"ema", "step", "elapsed"} and torch.equal(out["ema"]["w"], torch.zeros(2))


def test_policy_keeps_policy_and_counters():
    ck = {"policy": {"w": torch.ones(2)}, "opt": {"state": {}}, "it": 900, "elapsed": 2.0}
    assert set(strip_checkpoint(ck)) == {"policy", "it", "elapsed"}


def test_plain_state_dict_is_unchanged():
    sd = {"conv.weight": torch.ones(1), "conv.bias": torch.zeros(1)}
    assert strip_checkpoint(sd) is sd
