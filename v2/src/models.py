import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A standard convolutional residual block."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class Encoder(nn.Module):
    """
    CNN Encoder with residual blocks.
    Input shape: (B, 3, 144, 160)
    Output shape: (B, 512)
    """
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        # Downsampling Conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # (32, 72, 80)
        self.res1 = ResidualBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # (64, 36, 40)
        self.res2 = ResidualBlock(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # (128, 18, 20)
        self.res3 = ResidualBlock(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # (256, 9, 10)
        self.res4 = ResidualBlock(256)
        
        self.fc = nn.Linear(256 * 9 * 10, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.res1(h)
        h = F.relu(self.conv2(h))
        h = self.res2(h)
        h = F.relu(self.conv3(h))
        h = self.res3(h)
        h = F.relu(self.conv4(h))
        h = self.res4(h)
        h = h.reshape(h.size(0), -1)
        embed = self.fc(h)
        return embed

class Decoder(nn.Module):
    """
    CNN Decoder with residual blocks.
    Input shape: (B, latent_dim) [where latent_dim = det_dim + stochastic_dim]
    Output shape: (B, 3, 144, 160)
    """
    def __init__(self, latent_dim: int = 512 + 1024):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 9 * 10)
        
        self.res4 = ResidualBlock(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # (128, 18, 20)
        self.res3 = ResidualBlock(128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # (64, 36, 40)
        self.res2 = ResidualBlock(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # (32, 72, 80)
        self.res1 = ResidualBlock(32)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)   # (3, 144, 160)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.fc(latent)
        h = h.reshape(h.size(0), 256, 9, 10)
        h = self.res4(h)
        h = F.relu(self.deconv4(h))
        h = self.res3(h)
        h = F.relu(self.deconv1(h))
        h = self.res2(h)
        h = F.relu(self.deconv2(h))
        h = self.res1(h)
        # Sigmoid to normalize outputs between 0 and 1
        x_recon = torch.sigmoid(self.deconv3(h))
        return x_recon

class RSSMCell(nn.Module):
    """
    A single recurrent step of the Recurrent State-Space Model.
    Supports deterministic updates and discrete categorical latent predictions.
    """
    def __init__(self, action_dim: int = 8, det_dim: int = 512, class_num: int = 32, category_num: int = 32):
        super().__init__()
        self.action_dim = action_dim
        self.det_dim = det_dim
        self.class_num = class_num
        self.category_num = category_num
        self.stoch_dim = category_num * class_num  # 32 * 32 = 1024

        # GRU Cell for deterministic state update
        self.gru_cell = nn.GRUCell(self.stoch_dim + action_dim, det_dim)

        # Prior network: h_t -> p(s_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(det_dim, 512),
            nn.ELU(),
            nn.Linear(512, self.stoch_dim)
        )

        # Posterior network: [h_t, e_t] -> q(s_t | h_t, e_t)
        self.post_net = nn.Sequential(
            nn.Linear(det_dim + 512, 512),
            nn.ELU(),
            nn.Linear(512, self.stoch_dim)
        )

    def get_initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns zero-initialized deterministic state and uniform stochastic state."""
        h = torch.zeros(batch_size, self.det_dim, device=device)
        s = torch.zeros(batch_size, self.stoch_dim, device=device)
        # Initialize as uniform distribution representation (1/class_num)
        s = s.reshape(batch_size, self.category_num, self.class_num)
        s = F.softmax(s, dim=-1)
        s = s.reshape(batch_size, self.stoch_dim)
        return h, s

    def sample_stochastic(self, logits: torch.Tensor, use_gumbel: bool = True, hard: bool = True, temp: float = 1.0) -> torch.Tensor:
        """Sample discrete categorical latents using Gumbel-Softmax or straight-through argmax."""
        batch_size = logits.size(0)
        # Shape: (B, category_num, class_num)
        logits = logits.reshape(batch_size, self.category_num, self.class_num)
        
        if use_gumbel:
            # Straight-through Gumbel-Softmax
            sample = F.gumbel_softmax(logits, tau=temp, hard=hard)
        else:
            # Argmax with straight-through gradient copy
            probs = F.softmax(logits, dim=-1)
            indices = torch.argmax(probs, dim=-1)
            one_hot = F.one_hot(indices, num_classes=self.class_num).float()
            sample = one_hot + probs - probs.detach()  # straight-through gradient
            
        return sample.reshape(batch_size, self.stoch_dim)

    def forward(self, prev_h: torch.Tensor, prev_s: torch.Tensor, action: torch.Tensor, embed: torch.Tensor = None, use_gumbel: bool = True, temp: float = 1.0) -> dict:
        """
        Single RSSM step.
        prev_h: (B, det_dim)
        prev_s: (B, stoch_dim)
        action: (B, action_dim) one-hot action
        embed: (B, 512) optional encoder embedding (only available during training/inference with real frames)
        """
        # 1. Deterministic update
        gru_input = torch.cat([prev_s, action], dim=-1)
        h = self.gru_cell(gru_input, prev_h)

        # 2. Prior distribution prediction
        prior_logits = self.prior_net(h)
        
        # 3. Posterior distribution prediction (if real observations are provided)
        if embed is not None:
            post_logits = self.post_net(torch.cat([h, embed], dim=-1))
            # Sample from posterior during training
            s = self.sample_stochastic(post_logits, use_gumbel=use_gumbel, hard=True, temp=temp)
        else:
            post_logits = prior_logits.clone()
            # Sample from prior during generation/imagination
            s = self.sample_stochastic(prior_logits, use_gumbel=use_gumbel, hard=True, temp=temp)

        return {
            "h": h,
            "s": s,
            "prior_logits": prior_logits,
            "post_logits": post_logits
        }

class RewardPredictor(nn.Module):
    """Predicts scaler rewards from the latent state."""
    def __init__(self, latent_dim: int = 512 + 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)

class ContinuePredictor(nn.Module):
    """Predicts a Bernoulli probability of whether the episode continues (for discount/done)."""
    def __init__(self, latent_dim: int = 512 + 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(latent).squeeze(-1))

class Actor(nn.Module):
    """Policy network outputting logits over environment actions."""
    def __init__(self, latent_dim: int = 512 + 1024, action_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)

class Critic(nn.Module):
    """Value network outputting expected state returns."""
    def __init__(self, latent_dim: int = 512 + 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)
