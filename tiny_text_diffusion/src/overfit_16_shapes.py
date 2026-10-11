
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
DATA_DIR = PROJECT / "dataset/synthetic_shapes"
OUT_DIR = PROJECT / "checkpoints/overfit_16"
LOG_DIR = PROJECT / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

T = 50
EPOCHS = 100
BATCH_SIZE = 4
LR = 2e-4
SEED = 123
DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class Fixed16(Dataset):
    def __init__(self):
        with open(DATA_DIR / "labels.csv", newline="") as f:
            self.rows = list(csv.DictReader(f))[:16]

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
        self.down = nn.Conv2d(channels, channels * 2, 4, 2, 1)
        self.mid = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.up = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)
        self.out = nn.Conv2d(channels * 2, 3, 3, padding=1)

    def forward(self, x, t):
        a = F.silu(self.conv1(x))
        emb = self.time(t.float().view(-1, 1) / T)
        b = F.silu(self.down(a) + emb[:, :, None, None])
        b = F.silu(self.mid(b))
        c = F.silu(self.up(b))
        return self.out(torch.cat([a, c], dim=1))

def main():
    dataset = Fixed16()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Cosine schedule, identical definition to the earlier experiment.
    steps = torch.arange(T + 1, dtype=torch.float32)
    s = 0.008
    alpha_bar_all = torch.cos(
        ((steps / T + s) / (1 + s)) * torch.pi / 2
    ) ** 2
    alpha_bar_all = alpha_bar_all / alpha_bar_all[0]
    betas = (1 - alpha_bar_all[1:] / alpha_bar_all[:-1]).clamp(1e-5, 0.999)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    model = TinyUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    log_path = LOG_DIR / "overfit_16_loss.csv"
    checkpoint = OUT_DIR / "overfit_16_latest.pth"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "average_loss"])

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            for x in loader:
                x = x.to(DEVICE)
                t = torch.randint(0, T, (x.size(0),))
                noise = torch.randn_like(x)
                a = alpha_bars[t].view(-1, 1, 1, 1)
                noisy = a.sqrt() * x + (1 - a).sqrt() * noise

                loss = F.mse_loss(model(noisy, t), noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / len(loader)
            writer.writerow([epoch + 1, f"{avg:.6f}"])
            f.flush()

            saved = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "timesteps": T,
                "schedule": "cosine",
                "betas": betas,
                "alphas": alphas,
                "alpha_bars": alpha_bars,
            }
            torch.save(saved, checkpoint)

            if epoch + 1 in (25, 50, 75, 100):
                torch.save(saved, OUT_DIR / f"overfit_16_epoch_{epoch + 1}.pth")

            print(f"Epoch {epoch + 1:03d}/{EPOCHS} | loss={avg:.4f}", flush=True)

    print("Checkpoint:", checkpoint)
    print("Loss log:", log_path)

if __name__ == "__main__":
    main()
