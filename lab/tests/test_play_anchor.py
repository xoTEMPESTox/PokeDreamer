"""Tests for play.py re-anchoring helper logic."""
import pytest
from lab.env import BUTTONS
from lab.play import (
    ACTION_NAMES,
    anchor_status_text,
    is_anchor_step,
    next_anchor_in,
)


def test_button_names_match_env():
    for idx, name in ACTION_NAMES.items():
        assert name.lower() in BUTTONS


def test_is_anchor_step():
    assert not is_anchor_step(0, 4)
    assert not is_anchor_step(1, 4)
    assert not is_anchor_step(2, 4)
    assert not is_anchor_step(3, 4)
    assert is_anchor_step(4, 4)
    assert not is_anchor_step(5, 4)
    assert is_anchor_step(8, 4)
    assert is_anchor_step(12, 4)
    assert is_anchor_step(16, 4)
    assert not is_anchor_step(19, 4)


def test_19_presses_schedule():
    # 19 presses with N=4 must yield exactly 4 anchors at steps 4, 8, 12, 16
    anchors = [s for s in range(1, 20) if is_anchor_step(s, 4)]
    assert anchors == [4, 8, 12, 16]
    assert len(anchors) == 4


def test_anchor_off():
    # When anchor is 0 (off), no steps are anchor steps
    assert not any(is_anchor_step(s, 0) for s in range(30))
    assert next_anchor_in(5, 0) == 0
    assert anchor_status_text(5, 0) == ""


def test_next_anchor_in():
    assert next_anchor_in(0, 4) == 4
    assert next_anchor_in(1, 4) == 3
    assert next_anchor_in(2, 4) == 2
    assert next_anchor_in(3, 4) == 1
    assert next_anchor_in(4, 4) == 4
    assert next_anchor_in(5, 4) == 3


def test_anchor_status_text():
    assert anchor_status_text(0, 4) == "Anchor every 4 | next anchor in 4"
    assert anchor_status_text(1, 4) == "Anchor every 4 | next anchor in 3"
    assert anchor_status_text(3, 4) == "Anchor every 4 | next anchor in 1"
    assert anchor_status_text(4, 4) == "Anchor every 4 | ANCHORED"
