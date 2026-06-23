import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for compressing Pokémon Red observations.
    Input shape: (Batch, 3, 36, 40)
    Output shape: (Batch, 3, 36, 40)
    """
    def __init__(self, latent_dim: int = 32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder layers
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # (32, 18, 20)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # (64, 9, 10)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # (128, 5, 5)
        
        self.fc_mu = nn.Linear(128 * 5 * 5, latent_dim)
        self.fc_logvar = nn.Linear(128 * 5 * 5, latent_dim)

        # Decoder layers
        self.dec_fc = nn.Linear(latent_dim, 128 * 5 * 5)
        
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)) # (64, 9, 10)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1))  # (32, 18, 20)
        self.dec_conv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=(1, 1))   # (3, 36, 40)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input image tensor to latents.
        x: (B, 3, 36, 40) normalized to [0, 1]
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = h.reshape(h.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z = mu + std * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector z to reconstruct image.
        z: (B, latent_dim)
        Returns: (B, 3, 36, 40) normalized [0, 1]
        """
        h = F.relu(self.dec_fc(z))
        h = h.reshape(h.size(0), 128, 5, 5)
        
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        # Sigmoid to ensure output is in range [0, 1]
        x_recon = torch.sigmoid(self.dec_conv3(h))
        return x_recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, 3, 36, 40)
        Returns: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(self, x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> dict:
        """
        Computes VAE loss = Reconstruction Loss + beta * KL Divergence
        """
        # Reconstruction loss (BCE matches binary/gray pixels normalized between 0 and 1)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL Divergence loss: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        loss = recon_loss + beta * kl_loss
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
