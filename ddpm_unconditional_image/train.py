import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from dataset import get_dataloader
from diffusion import Diffusion
from model import UNet


# -------------------------
# Configuration
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-4


print("Using device:", device)


# -------------------------
# Load dataset
# -------------------------

dataloader = get_dataloader(
    batch_size=BATCH_SIZE
)


# -------------------------
# Create model
# -------------------------

model = UNet().to(device)


# -------------------------
# Diffusion
# -------------------------

diffusion = Diffusion(
    noise_steps=1000,
    img_size=32,
    device=device
)


# -------------------------
# Optimizer
# -------------------------

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# Loss
# -------------------------

mse = nn.MSELoss()


# -------------------------
# Training
# -------------------------

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    progress_bar = tqdm(dataloader)

    for images, labels in progress_bar:

        images = images.to(device)

        # Number of images in batch
        n = images.shape[0]

        # Pick random timesteps
        t = diffusion.sample_timesteps(n)

        # Add noise
        noisy_images, noise = diffusion.noise_images(
            images,
            t
        )

        # Predict noise
        predicted_noise = model(
            noisy_images,
            t
        )

        # Compare predicted noise
        # with actual noise
        loss = mse(
            predicted_noise,
            noise
        )

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model
        optimizer.step()

        progress_bar.set_postfix(
            loss=loss.item()
        )


# -------------------------
# Save trained model
# -------------------------

torch.save(
    model.state_dict(),
    "./outputs/ddpm_model.pth"
)

print("\nTraining complete!")
print("Model saved to outputs/ddpm_model.pth")