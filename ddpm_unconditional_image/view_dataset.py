import matplotlib.pyplot as plt

from dataset import get_dataloader


# Load a batch
dataloader = get_dataloader(batch_size=16)

images, labels = next(iter(dataloader))


# Convert [-1, 1] back to [0, 1]
images = (images + 1) / 2


# Display images
plt.figure(figsize=(10, 10))

for i in range(16):

    image = images[i]

    # CHW → HWC
    image = image.permute(1, 2, 0)

    plt.subplot(4, 4, i + 1)
    plt.imshow(image)

    plt.title(f"Class: {labels[i].item()}")
    plt.axis("off")


plt.tight_layout()

plt.savefig("./outputs/dataset_samples.png")

plt.show()