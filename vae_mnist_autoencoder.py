import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# --------------------------------
# 1. Load MNIST
# --------------------------------

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)


# --------------------------------
# 2. Define VAE
# --------------------------------

class VAE(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU()
        )

        # Mean of latent distribution
        self.fc_mu = nn.Linear(128, 2)

        # Log variance of latent distribution
        self.fc_logvar = nn.Linear(128, 2)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    # --------------------------------
    # Reparameterization trick
    # --------------------------------

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        epsilon = torch.randn_like(std)

        z = mu + epsilon * std

        return z

    # --------------------------------
    # Forward pass
    # --------------------------------

    def forward(self, x):

        h = self.encoder(x)

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)

        reconstructed = self.decoder(z)

        return reconstructed, mu, logvar


# --------------------------------
# 3. Create model
# --------------------------------

model = VAE()

print(model)


# --------------------------------
# 4. Optimizer
# --------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)