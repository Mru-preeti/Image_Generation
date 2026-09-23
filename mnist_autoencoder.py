from torchvision import datasets, transforms
from torch.utils.data import DataLoader
dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=
    transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
)
import torch
import torch.nn as nn

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):

        z = self.encoder(x)

        reconstructed = self.decoder(z)

        return reconstructed
print("Number of images:", len(dataset))

image, label = dataset[0]
print(image.shape)
print(image)
print("Image shape:", image.shape)
print("Label:", label)
print("Number of values:", image.numel())
import matplotlib.pyplot as plt

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Digit: {label}")
plt.axis("off")
plt.show()
model = Autoencoder()

print(model)
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(10):

    total_loss = 0

    for images, labels in loader:

        images = images.view(images.size(0), 784)

        reconstructed = model(images)

        loss = criterion(reconstructed, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {total_loss / len(loader):.4f}"
    )
    image, label = dataset[0]

# Turn 28×28 into 784
image_flat = image.view(1, 784)

# Send image through ONLY the encoder
with torch.no_grad():
    z = model.encoder(image_flat)

print("Digit:", label)
print("Latent point:", z)
with torch.no_grad():
    reconstruction = model.decoder(z)

reconstruction = reconstruction.view(28, 28)
plt.imshow(reconstruction, cmap="gray")
plt.title("Original latent point")
plt.axis("off")
plt.show()
# Original latent point
nearby_z = z + torch.tensor([[2.0, 1.0]])

print("Original z:", z)
print("Nearby z:", nearby_z)
with torch.no_grad():

    original_reconstruction = model.decoder(z)
    nearby_reconstruction = model.decoder(nearby_z)

original_reconstruction = original_reconstruction.view(28, 28)
nearby_reconstruction = nearby_reconstruction.view(28, 28)
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_reconstruction, cmap="gray")
plt.title("Original latent point")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(nearby_reconstruction, cmap="gray")
plt.title("Nearby latent point")
plt.axis("off")

plt.show()
all_z = []

with torch.no_grad():

    for images, labels in loader:

        images = images.view(images.size(0), 784)

        z_batch = model.encoder(images)

        all_z.append(z_batch)

all_z = torch.cat(all_z)

print("Minimum:", all_z.min(dim=0).values)
print("Maximum:", all_z.max(dim=0).values)
# Try a point outside the learned latent space
random_z = torch.tensor([[30.0, 30.0]])

# Decode it
generated = model.decoder(random_z)

# Reshape back to image
generated_image = generated.reshape(28, 28)

# Show it
plt.imshow(generated_image.detach().numpy(), cmap="gray")
plt.title("Decoded from outside latent space")
plt.axis("off")
plt.show()