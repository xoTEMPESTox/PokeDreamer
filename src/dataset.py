import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class PokemonDataset(Dataset):
    """
    Dataset for loading Pokémon Red gameplay transitions from NPZ files.
    Supports returning individual frames (for VAE) or trajectory sequences (for GRU/RNN).
    """
    def __init__(self, data_dir: str | Path, seq_len: int = 1, transform = None):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.transform = transform

        # Find all NPZ files
        self.files = sorted(list(self.data_dir.glob("transitions_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No transitions_*.npz files found in {data_dir}")

        print(f"[PokemonDataset] Loading {len(self.files)} files from {data_dir}...")
        
        # Load and concatenate all buffers into memory for speed
        all_obs = []
        all_actions = []
        all_rewards = []
        all_episode_starts = []
        all_map_ids = []
        all_xs = []
        all_ys = []
        all_facings = []
        all_in_battles = []
        all_dialog_opens = []
        all_badges = []
        all_party_hps = []
        all_party_max_hps = []

        for f in self.files:
            data = np.load(f)
            all_obs.append(data['obs'])
            all_actions.append(data['actions'])
            all_rewards.append(data['rewards'])
            all_episode_starts.append(data['episode_starts'])
            
            # Load RAM fields if they exist (for probe/eval)
            if 'map_ids' in data:
                all_map_ids.append(data['map_ids'])
                all_xs.append(data['xs'])
                all_ys.append(data['ys'])
                all_facings.append(data['facings'])
                all_in_battles.append(data['in_battles'])
                all_dialog_opens.append(data['dialog_opens'])
                all_badges.append(data['badges'])
                all_party_hps.append(data['party_hps'])
                all_party_max_hps.append(data['party_max_hps'])

        self.obs = np.concatenate(all_obs, axis=0) # (N, 36, 40, 3)
        self.actions = np.concatenate(all_actions, axis=0) # (N,)
        self.rewards = np.concatenate(all_rewards, axis=0) # (N,)
        self.episode_starts = np.concatenate(all_episode_starts, axis=0) # (N,)

        self.has_ram = len(all_map_ids) > 0
        if self.has_ram:
            self.map_ids = np.concatenate(all_map_ids, axis=0)
            self.xs = np.concatenate(all_xs, axis=0)
            self.ys = np.concatenate(all_ys, axis=0)
            self.facings = np.concatenate(all_facings, axis=0)
            self.in_battles = np.concatenate(all_in_battles, axis=0)
            self.dialog_opens = np.concatenate(all_dialog_opens, axis=0)
            self.badges = np.concatenate(all_badges, axis=0)
            self.party_hps = np.concatenate(all_party_hps, axis=0)
            self.party_max_hps = np.concatenate(all_party_max_hps, axis=0)

        self.num_total = len(self.obs)
        print(f"[PokemonDataset] Loaded {self.num_total} total frames (RAM fields: {self.has_ram})")

        # Pre-compute valid starting indices for sequence sequences
        if self.seq_len > 1:
            self.valid_indices = []
            # An index i is valid if the sequence [i, i + seq_len] does not cross a reset/start boundary
            for i in range(self.num_total - self.seq_len):
                # If there's an episode start in the middle of the sequence, it's invalid
                if not np.any(self.episode_starts[i + 1 : i + self.seq_len]):
                    self.valid_indices.append(i)
            print(f"[PokemonDataset] Found {len(self.valid_indices)} valid sequences of length {self.seq_len}")
        else:
            self.valid_indices = list(range(self.num_total))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        start_idx = self.valid_indices[idx]
        
        if self.seq_len == 1:
            # VAE mode: return a single normalized frame and metadata
            img = self.obs[start_idx] # (36, 40, 3)
            # Rearrange channel to PyTorch format (3, 36, 40)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
            
            sample = {
                'obs': torch.tensor(img, dtype=torch.float32),
                'action': int(self.actions[start_idx])
            }
            if self.has_ram:
                sample.update({
                    'map_id': int(self.map_ids[start_idx]),
                    'x': int(self.xs[start_idx]),
                    'y': int(self.ys[start_idx]),
                    'facing': int(self.facings[start_idx]),
                    'in_battle': bool(self.in_battles[start_idx]),
                    'dialog_open': bool(self.dialog_opens[start_idx]),
                    'badges': int(self.badges[start_idx]),
                    'party_hp': torch.tensor(self.party_hps[start_idx], dtype=torch.float32),
                    'party_max_hp': torch.tensor(self.party_max_hps[start_idx], dtype=torch.float32)
                })
            return sample
        else:
            # Trajectory sequence mode (for dynamics training)
            end_idx = start_idx + self.seq_len
            
            # Rearrange shape from (T, H, W, C) to (T, C, H, W)
            imgs = self.obs[start_idx:end_idx]
            imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255.0
            
            sample = {
                'obs': torch.tensor(imgs, dtype=torch.float32), # (T, C, H, W)
                'actions': torch.tensor(self.actions[start_idx:end_idx], dtype=torch.long), # (T,)
                'rewards': torch.tensor(self.rewards[start_idx:end_idx], dtype=torch.float32) # (T,)
            }
            
            if self.has_ram:
                sample.update({
                    'map_ids': torch.tensor(self.map_ids[start_idx:end_idx], dtype=torch.long),
                    'xs': torch.tensor(self.xs[start_idx:end_idx], dtype=torch.float32),
                    'ys': torch.tensor(self.ys[start_idx:end_idx], dtype=torch.float32),
                    'facings': torch.tensor(self.facings[start_idx:end_idx], dtype=torch.long),
                    'in_battles': torch.tensor(self.in_battles[start_idx:end_idx], dtype=torch.float32),
                    'dialog_opens': torch.tensor(self.dialog_opens[start_idx:end_idx], dtype=torch.float32),
                    'badges': torch.tensor(self.badges[start_idx:end_idx], dtype=torch.float32),
                    'party_hps': torch.tensor(self.party_hps[start_idx:end_idx], dtype=torch.float32),
                    'party_max_hps': torch.tensor(self.party_max_hps[start_idx:end_idx], dtype=torch.float32)
                })
            return sample
