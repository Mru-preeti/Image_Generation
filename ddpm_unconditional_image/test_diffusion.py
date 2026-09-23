import torch
import matplotlib.pyplot as plt

from dataset import get_dataloader
from diffusion import Diffusion


# -------------------------
# Device
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# -------------------------
# Load dataset
# -------------------------
dataloader = get_dataloader(batch_size=1)

images, labels = next(iter(dataloader))

images = images.to(device)


# -------------------------
# Create diffusion object
# -------------------------
diffusion = Diffusion(
    noise_steps=1000,
    img_size=32,
    device=device
)


# -------------------------
# Timesteps to visualize
# -------------------------
timesteps = [1, 100, 300, 600, 999]


# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(15, 3))

for i, timestep in enumerate(timesteps):

    t = torch.tensor(
        [timestep],
        device=device
    )

    noisy_image, noise = diffusion.noise_images(
        images,
        t
    )

    # Convert from [-1, 1] to [0, 1]
    noisy_image = (
        noisy_image.clamp(-1, 1) + 1
    ) / 2

    noisy_image = noisy_image.squeeze(0)
    noisy_image = noisy_image.permute(1, 2, 0)

    plt.subplot(1, len(timesteps), i + 1)

    plt.imshow(noisy_image.cpu())

    plt.title(f"t = {timestep}")
    plt.axis("off")


plt.tight_layout()

plt.savefig(
    "./outputs/forward_diffusion.png"
)

plt.show()