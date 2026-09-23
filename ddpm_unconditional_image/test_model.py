import torch

from model import UNet


device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)

print("Using device:", device)


# Fake noisy images
x = torch.randn(
    4,
    3,
    32,
    32
).to(device)


# Random timesteps
t = torch.randint(
    1,
    1000,
    (4,)
).to(device)


# U-Net prediction
output = model(x, t)


print("Input shape: ", x.shape)
print("Output shape:", output.shape)