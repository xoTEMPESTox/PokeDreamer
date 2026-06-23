import random
import torch
import torch.nn as nn

class LatentDynamics(nn.Module):
    """
    Recurrent Latent Dynamics Model (RNN/GRU transition model) for PokéWorld v3.
    Learns to predict: (z_t, action_t, hidden_t) -> (z_{t+1}, hidden_{t+1})
    """
    def __init__(
        self, 
        latent_dim: int = 32, 
        num_actions: int = 8, 
        action_dim: int = 16, 
        hidden_dim: int = 256
    ):
        super(LatentDynamics, self).__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Embed actions to continuous space
        self.action_embed = nn.Embedding(num_actions, action_dim)
        
        # Recurrent cell
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        
        # Predict next latent state z_{t+1} from hidden state
        self.fc_next_z = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Predict reward
        self.fc_reward = nn.Linear(hidden_dim, 1)

    def forward(
        self, 
        z_seq: torch.Tensor, 
        action_seq: torch.Tensor, 
        teacher_forcing_ratio: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence of latents and actions with scheduled sampling.
        
        Parameters
        ----------
        z_seq : torch.Tensor
            Ground truth latents from VAE, shape (B, T, latent_dim)
        action_seq : torch.Tensor
            Actions taken, shape (B, T)
        teacher_forcing_ratio : float
            Probability of using ground truth z_t instead of predicted z_t at each step.
            
        Returns
        -------
        predicted_z_seq : torch.Tensor
            Predicted next latents z_{t+1} for t=0..T-1, shape (B, T, latent_dim)
        predicted_rewards : torch.Tensor
            Predicted rewards, shape (B, T)
        """
        B, T, _ = z_seq.size()
        device = z_seq.device
        
        # Embed actions: (B, T, action_dim)
        action_embeds = self.action_embed(action_seq)
        
        # Initialize GRU hidden state to zeros
        hidden = torch.zeros(B, self.hidden_dim, device=device)
        
        predicted_z_list = []
        predicted_reward_list = []
        
        # First step input is the first ground truth latent
        current_z = z_seq[:, 0]
        
        for t in range(T):
            act_t = action_embeds[:, t]
            
            # Concatenate latent state and action embedding
            gru_input = torch.cat([current_z, act_t], dim=-1)
            
            # Recurrent step
            hidden = self.gru(gru_input, hidden)
            
            # Predict next latent state z_{t+1} and reward
            pred_next_z = self.fc_next_z(hidden)
            pred_reward = self.fc_reward(hidden).squeeze(-1)
            
            predicted_z_list.append(pred_next_z)
            predicted_reward_list.append(pred_reward)
            
            # Prepare input for next step
            if t < T - 1:
                # Scheduled sampling: choose between ground truth and model's own prediction
                if teacher_forcing_ratio >= 1.0 or (teacher_forcing_ratio > 0.0 and random.random() < teacher_forcing_ratio):
                    current_z = z_seq[:, t + 1]
                else:
                    # Let gradients flow through predictions to allow learning self-correction
                    current_z = pred_next_z
                    
        predicted_z_seq = torch.stack(predicted_z_list, dim=1)
        predicted_rewards = torch.stack(predicted_reward_list, dim=1)
        
        return predicted_z_seq, predicted_rewards

    def rollout(
        self, 
        z_start: torch.Tensor, 
        action_sequence: list[int] | torch.Tensor, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Roll out imagined futures purely in latent space.
        
        Parameters
        ----------
        z_start : torch.Tensor
            Starting latent state, shape (B, latent_dim) or (latent_dim,)
        action_sequence : list[int] or torch.Tensor
            Sequence of actions to apply, shape (T,) or (B, T)
            
        Returns
        -------
        torch.Tensor
            Sequence of imagined future latent states, shape (B, T, latent_dim)
        """
        self.eval()
        with torch.no_grad():
            # Handle shape variants
            if len(z_start.shape) == 1:
                z_start = z_start.unsqueeze(0) # (1, latent_dim)
            
            B = z_start.size(0)
            
            if isinstance(action_sequence, list):
                action_seq = torch.tensor(action_sequence, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1) # (B, T)
            else:
                action_seq = action_sequence
                if len(action_seq.shape) == 1:
                    action_seq = action_seq.unsqueeze(0) # (1, T)
            
            T = action_seq.size(1)
            
            # Embed actions
            action_embeds = self.action_embed(action_seq)
            
            hidden = torch.zeros(B, self.hidden_dim, device=device)
            current_z = z_start
            
            imagined_z_list = []
            
            for t in range(T):
                act_t = action_embeds[:, t]
                gru_input = torch.cat([current_z, act_t], dim=-1)
                hidden = self.gru(gru_input, hidden)
                
                # Predict next latent
                current_z = self.fc_next_z(hidden)
                imagined_z_list.append(current_z)
                
            return torch.stack(imagined_z_list, dim=1) # (B, T, latent_dim)
