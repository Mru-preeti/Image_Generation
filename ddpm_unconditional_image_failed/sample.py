import torch

from model import UNet
from diffusion import Diffusion


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# Load trained U-Net
model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "./outputs/ddpm_model.pth",
        map_location=device
    )
)

model.eval()


# Create diffusion process
diffusion = Diffusion(
    noise_steps=1000,
    img_size=32,
    device=device
)


# Start with random noise
x = torch.randn(
    1,
    3,
    32,
    32
).to(device)


print("Starting noise shape:", x.shape)
# Start reverse diffusion
for i in range(999, 0, -1):

    t = torch.tensor([i], device=device)

    # U-Net predicts the noise
    predicted_noise = model(x, t)

    # Get beta, alpha and alpha_hat for this timestep
    beta = diffusion.beta[t][:, None, None, None]
    alpha = diffusion.alpha[t][:, None, None, None]
    alpha_hat = diffusion.alpha_hat[t][:, None, None, None]

    # Random noise for the sampling step
    if i > 1:
        noise = torch.randn_like(x)
    else:
        noise = torch.zeros_like(x)

    # Reverse diffusion equation
    x = (
        1 / torch.sqrt(alpha)
    ) * (
        x
        - ((1 - alpha) / torch.sqrt(1 - alpha_hat))
        * predicted_noise
    ) + torch.sqrt(beta) * noise

    # Keep image values within range
    x = x.clamp(-1, 1)

print("Sampling complete!")
import matplotlib.pyplot as plt


# Convert from [-1, 1] to [0, 1]
x = (x.clamp(-1, 1) + 1) / 2

# Remove batch dimension
image = x[0]

# Convert CHW → HWC
image = image.permute(1, 2, 0)

# Move from GPU/CPU tensor to NumPy
#image = image.cpu().numpy()
image = image.detach().cpu().numpy()

# Display image
plt.imshow(image)
plt.axis("off")
plt.title("Generated Image")

plt.savefig(
    "./outputs/generated.png"
)

plt.show()