
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = Path("/content/drive/MyDrive/tiny_text_diffusion")
DATA_DIR = PROJECT / "dataset" / "synthetic_shapes"
CHECKPOINT_DIR = PROJECT / "checkpoints" / "latest"
LOG_DIR = PROJECT / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 3
BATCH_SIZE = 16
TIMESTEPS = 50
DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class ShapeDataset(Dataset):
    def __init__(self):
        with open(DATA_DIR / "labels.csv", newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img = Image.open(DATA_DIR / self.rows[idx]["filename"]).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

class TinyUNet(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.time = nn.Sequential(
            nn.Linear(1, channels), nn.SiLU(),
            nn.Linear(channels, channels * 2)
        )
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.down = nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1)
        self.mid = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.up = nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1)
        self.out = nn.Conv2d(channels * 2, 3, 3, padding=1)

    def forward(self, x, t):
        a = F.silu(self.conv1(x))
        emb = self.time(t.float().view(-1, 1) / TIMESTEPS)
        b = F.silu(self.down(a) + emb[:, :, None, None])
        b = F.silu(self.mid(b))
        c = F.silu(self.up(b))
        return self.out(torch.cat([a, c], dim=1))

def main():
    dataset = ShapeDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    betas = torch.linspace(1e-4, 0.02, TIMESTEPS)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    model = TinyUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    checkpoint = CHECKPOINT_DIR / "shape_diffusion_latest.pth"
    log_path = LOG_DIR / "training_loss.csv"

    with open(log_path, "w") as log:
        log.write("epoch,average_loss\n")
        for epoch in range(EPOCHS):
            model.train()
            total = 0.0

            for x in loader:
                x = x.to(DEVICE)
                t = torch.randint(0, TIMESTEPS, (x.size(0),))
                noise = torch.randn_like(x)
                a = alpha_bars[t].view(-1, 1, 1, 1)
                noisy = a.sqrt() * x + (1 - a).sqrt() * noise

                loss = F.mse_loss(model(noisy, t), noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()

            average = total / len(loader)
            log.write(f"{epoch + 1},{average:.6f}\n")
            log.flush()

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "timesteps": TIMESTEPS,
                "seed": SEED
            }, checkpoint)

            print(f"Epoch {epoch + 1}/{EPOCHS}: loss={average:.4f}")

    print("Checkpoint:", checkpoint)
    print("Loss log:", log_path)

if __name__ == "__main__":
    main()
