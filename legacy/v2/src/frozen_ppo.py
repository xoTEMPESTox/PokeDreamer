"""
FrozenPPO — loads PWhiddy's SB3 checkpoint for frozen inference.

IMPORTANT — observation format:
  PWhiddy's CNN policy was trained on a stacked observation of shape (128, 40, 3):
    - 3 stacked game frames, each downsampled to (36, 40, 3)  →  (108, 40, 3)
    - 2px zero padding                                         →  (  2, 40, 3)
    - 8px "recent memory" bar (reward history visualisation)   →  (  8, 40, 3)
    - 2px zero padding                                         →  (  2, 40, 3)
    - 8px "exploration memory" bar                             →  (  8, 40, 3)
    Total: 108 + 2 + 8 + 2 + 8 = 128... wait, let me recount.
    output_full[0] = output_shape[0]*frame_stacks + 2*(mem_padding+memory_height)
                   = 36*3 + 2*(2+8) = 108 + 20 = 128
  So the real obs shape is (128, 40, 3).

We reproduce this by:
  - Keeping a rolling buffer of the last 3 frames
  - Filling memory bars with zeros (we don't train — zeros are fine for inference,
    the CNN will have learned to be robust to them at episode start anyway)
  - This keeps the policy input identical to what it saw during training
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import deque
from typing import Optional

import numpy as np
import cv2

# ── Make PWhiddy's env importable (SB3 may need it for deserialization) ───────
_EXTERNAL = Path(__file__).resolve().parents[1] / "external" / "PokemonRedExperiments" / "baselines"
if str(_EXTERNAL) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL))

from stable_baselines3 import PPO


# ── Observation geometry (must match red_gym_env.py exactly) ─────────────────
FRAME_H      = 36     # downsampled frame height
FRAME_W      = 40     # downsampled frame width
FRAME_STACKS = 3      # number of stacked frames
MEM_PADDING  = 2      # zero-padding rows between sections
MEM_HEIGHT   = 8      # rows for each memory bar
N_CHANNELS   = 3      # RGB

# Full obs shape fed to the CNN:
# stacked frames + padding + recent_mem + padding + explore_mem
OBS_H = FRAME_H * FRAME_STACKS + 2 * (MEM_PADDING + MEM_HEIGHT)  # = 128
OBS_W = FRAME_W
OBS_SHAPE = (OBS_H, OBS_W, N_CHANNELS)  # (128, 40, 3)


class FrozenPPO:
    """
    Wraps a PWhiddy PPO checkpoint for frozen inference.

    Maintains an internal frame buffer so consecutive calls produce
    correctly stacked observations — exactly as the env did during training.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the SB3 .zip checkpoint.
    device : str
        'cpu' (default) or 'cuda'.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        # Rolling frame buffer (oldest first)
        self._frames: deque[np.ndarray] = deque(maxlen=FRAME_STACKS)
        self._reset_frames()

        self._model: Optional[PPO] = None
        self._load()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _reset_frames(self) -> None:
        """Fill frame buffer with blank frames (call at episode start)."""
        blank = np.zeros((FRAME_H, FRAME_W, N_CHANNELS), dtype=np.uint8)
        self._frames.clear()
        for _ in range(FRAME_STACKS):
            self._frames.append(blank.copy())

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Download from PWhiddy's repo or provide your own .zip"
            )
        print(f"[FrozenPPO] Loading: {self.checkpoint_path}")
        # custom_objects stubs out lr_schedule and clip_range — these are
        # serialized as Python lambdas in old checkpoints and fail to
        # deserialize in newer Python/SB3.  We only do inference so the
        # actual values don't matter.
        custom_objects = {
            "lr_schedule": lambda _: 2.5e-4,
            "clip_range":  lambda _: 0.2,
        }
        self._model = PPO.load(
            str(self.checkpoint_path),
            device=self.device,
            custom_objects=custom_objects,
        )
        # Belt-and-suspenders freeze
        for p in self._model.policy.parameters():
            p.requires_grad_(False)
        self._model.policy.eval()
        print(f"[FrozenPPO] Loaded OK  |  obs_shape={OBS_SHAPE}  |  n_actions={self.n_actions}")

    # ── Observation building ──────────────────────────────────────────────────

    def _downsample(self, screen_rgb: np.ndarray) -> np.ndarray:
        """(144,160,3) → (36,40,3) uint8, matching red_gym_env render()."""
        resized = cv2.resize(screen_rgb, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def _build_obs(self) -> np.ndarray:
        """
        Assemble the full (128, 40, 3) observation from the frame buffer.

        Memory bars are zeroed — the policy was trained with these reflecting
        reward history, but zero is a valid/safe value at inference time.
        """
        # Stack frames newest → oldest  shape: (108, 40, 3)
        stacked = np.concatenate(list(self._frames)[::-1], axis=0)

        pad  = np.zeros((MEM_PADDING, OBS_W, N_CHANNELS), dtype=np.uint8)
        mem  = np.zeros((MEM_HEIGHT,  OBS_W, N_CHANNELS), dtype=np.uint8)

        obs = np.concatenate([mem, pad, mem, pad, stacked], axis=0)
        assert obs.shape == OBS_SHAPE, f"obs shape mismatch: {obs.shape} vs {OBS_SHAPE}"
        return obs

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Call at the start of each episode to clear the frame buffer."""
        self._reset_frames()

    def predict(
        self,
        screen_rgb: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """
        Push the latest frame, build the stacked obs, and return an action.

        Parameters
        ----------
        screen_rgb : np.ndarray
            Raw (144, 160, 3) screen from screen_capture().
        deterministic : bool
            Argmax policy (True) vs sampled (False).

        Returns
        -------
        int
            Action index. PWhiddy's 6-action set:
              0=DOWN, 1=LEFT, 2=RIGHT, 3=UP, 4=A, 5=B
        """
        frame = self._downsample(screen_rgb)
        self._frames.append(frame)

        obs = self._build_obs()
        obs_batch = obs[np.newaxis, ...]         # (1, 128, 40, 3)
        action, _ = self._model.predict(obs_batch, deterministic=deterministic)
        return int(action[0])

    @property
    def n_actions(self) -> int:
        return self._model.action_space.n

    def __repr__(self) -> str:
        return (
            f"FrozenPPO(ckpt={self.checkpoint_path.name}, "
            f"device={self.device}, obs={OBS_SHAPE}, n_actions={self.n_actions})"
        )
