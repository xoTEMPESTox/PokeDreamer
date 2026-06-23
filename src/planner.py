import random
import torch
import torch.nn as nn
from src.dynamics import LatentDynamics
from src.probe import RAMProbe

class LatentPlanner:
    """
    Lookahead Planner that imagines futures using the LatentDynamics model,
    probes the imagined latent states using the RAMProbe, and selects the action sequence
    that maximizes a specified objective.
    """
    def __init__(self, dynamics: LatentDynamics, probe: RAMProbe, device: torch.device):
        self.dynamics = dynamics
        self.probe = probe
        self.device = device

    def generate_candidates(self, seq_len: int = 15) -> dict[str, list[int]]:
        """
        Generate a set of candidate macro-action sequences.
        Actions: 0=DOWN, 1=LEFT, 2=RIGHT, 3=UP, 4=A, 5=B
        """
        candidates = {}
        
        # 1. Straight directional macro-actions
        candidates["go_down"] = [0] * seq_len
        candidates["go_left"] = [1] * seq_len
        candidates["go_right"] = [2] * seq_len
        candidates["go_up"] = [3] * seq_len
        
        # 2. Diagonal-ish zigzag macro-actions
        zigzag_down_right = []
        zigzag_down_left = []
        zigzag_up_right = []
        zigzag_up_left = []
        for i in range(seq_len):
            zigzag_down_right.append(0 if i % 2 == 0 else 2)
            zigzag_down_left.append(0 if i % 2 == 0 else 1)
            zigzag_up_right.append(3 if i % 2 == 0 else 2)
            zigzag_up_left.append(3 if i % 2 == 0 else 1)
            
        candidates["zigzag_down_right"] = zigzag_down_right
        candidates["zigzag_down_left"] = zigzag_down_left
        candidates["zigzag_up_right"] = zigzag_up_right
        candidates["zigzag_up_left"] = zigzag_up_left
        
        # 3. Interactive/dialog macro-actions (pressing A or B repeatedly or mixed with directions)
        candidates["interact_a"] = [4] * seq_len
        candidates["interact_b"] = [5] * seq_len
        
        # A mixed sequence to try to clear text and walk down
        mixed_a_down = []
        for i in range(seq_len):
            mixed_a_down.append(4 if i % 3 == 0 else 0)
        candidates["mixed_a_down"] = mixed_a_down

        # 4. Random walk exploration sequences (for diverse search)
        for r in range(5):
            seq = [random.randint(0, 5) for _ in range(seq_len)]
            candidates[f"random_{r}"] = seq
            
        return candidates

    def evaluate_imagined_state(
        self, 
        pred_state: dict, 
        goal_x: int, 
        goal_y: int, 
        goal_map_id: int | None = None,
        avoid_battles: bool = True,
        avoid_dialog: bool = True
    ) -> float:
        """
        Evaluate and score a predicted game state from the probe.
        
        Parameters
        ----------
        pred_state : dict
            State prediction dictionary from RAMProbe.
        goal_x, goal_y : int
            Target coordinates.
        goal_map_id : int or None
            Target map ID (if None, assumes current map).
        avoid_battles : bool
            If True, penalize states predicted to be in battle.
        avoid_dialog : bool
            If True, penalize states predicted to have a dialog open.
            
        Returns
        -------
        score : float
        """
        # Extract predicted values
        pred_pos = pred_state['pos'].squeeze(0).cpu().numpy() # (2,) -> (x, y)
        pred_x, pred_y = pred_pos[0], pred_pos[1]
        
        # Battle & dialog flags (sigmoid threshold)
        pred_in_battle = torch.sigmoid(pred_state['battle_logits']).item() > 0.5
        pred_dialog_open = torch.sigmoid(pred_state['dialog_logits']).item() > 0.5
        
        pred_map_logits = pred_state['map_logits'].squeeze(0)
        pred_map_id = torch.argmax(pred_map_logits).item()
        
        score = 0.0
        
        # 1. Navigation Reward (negative Manhattan distance)
        manhattan_dist = abs(pred_x - goal_x) + abs(pred_y - goal_y)
        
        # If target map is specified
        if goal_map_id is not None:
            if pred_map_id == goal_map_id:
                # Big reward for reaching target map + proximity reward
                score += 1000.0 - manhattan_dist
            else:
                # Penalize wrong map, but reward moving closer in latent space if starting map is same
                score -= 1000.0
        else:
            # Assumed same map navigation
            score -= manhattan_dist
            
        # 2. Penalty constraints
        if avoid_battles and pred_in_battle:
            score -= 50.0  # penalize landing in battle
            
        if avoid_dialog and pred_dialog_open:
            score -= 10.0  # slight penalty for text boxes (encourages moving over talking if trying to travel)
            
        return float(score)

    def plan(
        self, 
        z_start: torch.Tensor, 
        goal_x: int, 
        goal_y: int, 
        goal_map_id: int | None = None,
        seq_len: int = 15,
        avoid_battles: bool = True
    ) -> tuple[str, list[int], float]:
        """
        Compare imagined futures and return the best action sequence.
        
        Parameters
        ----------
        z_start : torch.Tensor
            Current latent state, shape (latent_dim,) or (1, latent_dim).
        goal_x, goal_y, goal_map_id : int
            Targets.
        seq_len : int
            Lookahead steps.
            
        Returns
        -------
        best_name : str
            Name of the winning macro-action.
        best_sequence : list[int]
            Sequence of actions.
        best_score : float
            Imagined score of the winning sequence.
        """
        # Ensure batch shape (1, latent_dim)
        if len(z_start.shape) == 1:
            z_start = z_start.unsqueeze(0)
            
        candidates = self.generate_candidates(seq_len=seq_len)
        
        best_name = None
        best_sequence = None
        best_score = float("-inf")
        
        # Evaluate each sequence
        for name, seq in candidates.items():
            # Rollout in latent space
            # pred_z_seq: (1, seq_len, latent_dim)
            pred_z_seq = self.dynamics.rollout(z_start, seq, device=self.device)
            
            # Extract final predicted latent state z_{t+k}
            z_final = pred_z_seq[:, -1] # (1, latent_dim)
            
            # Probe final state
            pred_state = self.probe(z_final)
            
            # Score final state
            score = self.evaluate_imagined_state(
                pred_state, 
                goal_x=goal_x, 
                goal_y=goal_y, 
                goal_map_id=goal_map_id,
                avoid_battles=avoid_battles
            )
            
            if score > best_score:
                best_score = score
                best_name = name
                best_sequence = seq
                
        return best_name, best_sequence, best_score
