import torch
import torch.nn as nn
import torch.nn.functional as F

class RAMProbe(nn.Module):
    """
    Linear/MLP probes trained on VAE latent z to predict RAM states.
    Used to evaluate what information the latent space retains and for planning.
    """
    def __init__(self, latent_dim: int = 32, num_classes_map: int = 256):
        super(RAMProbe, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes_map = num_classes_map

        # Coordinate head (regression for x, y)
        self.pos_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # Map ID head (classification)
        self.map_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes_map)
        )
        
        # Status flag heads (binary classification logits)
        self.battle_head = nn.Linear(latent_dim, 1)
        self.dialog_head = nn.Linear(latent_dim, 1)
        
        # HP head (regression for 6 party HP fractions)
        self.hp_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, z: torch.Tensor) -> dict:
        """
        z: (B, latent_dim)
        Returns: dict of logits and regression predictions
        """
        pos = self.pos_head(z)       # (B, 2) -> (x, y)
        map_logits = self.map_head(z) # (B, num_classes_map)
        battle_logits = self.battle_head(z).squeeze(-1) # (B,)
        dialog_logits = self.dialog_head(z).squeeze(-1) # (B,)
        hp = self.hp_head(z)         # (B, 6)
        
        return {
            'pos': pos,
            'map_logits': map_logits,
            'battle_logits': battle_logits,
            'dialog_logits': dialog_logits,
            'hp': hp
        }

    def loss_function(self, preds: dict, targets: dict) -> dict:
        """
        Computes multi-task loss for the probes.
        """
        # 1. Position Loss (MSE)
        pos_loss = F.mse_loss(preds['pos'], targets['pos'])
        
        # 2. Map ID Loss (Cross Entropy)
        map_loss = F.cross_entropy(preds['map_logits'], targets['map_id'])
        
        # 3. Battle status Loss (BCE)
        battle_loss = F.binary_cross_entropy_with_logits(preds['battle_logits'], targets['in_battle'].float())
        
        # 4. Dialog status Loss (BCE)
        dialog_loss = F.binary_cross_entropy_with_logits(preds['dialog_logits'], targets['dialog_open'].float())
        
        # 5. HP Loss (MSE)
        hp_loss = F.mse_loss(preds['hp'], targets['hp'])
        
        total_loss = pos_loss + map_loss + 10.0 * battle_loss + 10.0 * dialog_loss + hp_loss
        
        return {
            'loss': total_loss,
            'pos_loss': pos_loss,
            'map_loss': map_loss,
            'battle_loss': battle_loss,
            'dialog_loss': dialog_loss,
            'hp_loss': hp_loss
        }
