
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
SOURCE_CKPT = PROJECT / "checkpoints/cosine_experiment/shape_diffusion_cosine_latest.pth"
OUT_DIR = PROJECT / "checkpoints/cosine_continuation"
LOG_DIR = PROJECT / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EPOCH = 100
BATCH_SIZE = 16
LR = 1e-4
DEVICE = torch.device("cpu")
SEED = 123

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
    def __init__(self, channels=16, timesteps=50):
        super().__init__()
        self.timesteps = timesteps
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
        emb = self.time(t.float().view(-1, 1) / self.timesteps)
        b = F.silu(self.down(a) + emb[:, :, None, None])
        b = F.silu(self.mid(b))
        c = F.silu(self.up(b))
        return self.out(torch.cat([a, c], dim=1))

def main():
    if not SOURCE_CKPT.exists():
        raise FileNotFoundError(f"Epoch-30 checkpoint missing: {SOURCE_CKPT}")

    ckpt = torch.load(SOURCE_CKPT, map_location=DEVICE, weights_only=True)
    start_epoch = int(ckpt["epoch"])
    if start_epoch >= TARGET_EPOCH:
        raise ValueError(f"Checkpoint already at epoch {start_epoch}.")

    T = int(ckpt["timesteps"])
    betas = ckpt["betas"].to(DEVICE)
    alphas = ckpt["alphas"].to(DEVICE)
    alpha_bars = ckpt["alpha_bars"].to(DEVICE)

    dataset = ShapeDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = TinyUNet(timesteps=T).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Use a smaller learning rate for fine-tuning.
    for group in optimizer.param_groups:
        group["lr"] = LR

    log_path = LOG_DIR / "training_loss_cosine_epochs31_100.csv"
    latest_path = OUT_DIR / "shape_diffusion_cosine_latest.pth"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "average_loss", "learning_rate"])

        for epoch in range(start_epoch, TARGET_EPOCH):
            model.train()
            total_loss = 0.0

            for x in loader:
                x = x.to(DEVICE)
                t = torch.randint(0, T, (x.size(0),), device=DEVICE)
                noise = torch.randn_like(x)
                a_bar = alpha_bars[t].view(-1, 1, 1, 1)
                noisy = a_bar.sqrt() * x + (1 - a_bar).sqrt() * noise

                prediction = model(noisy, t)
                loss = F.mse_loss(prediction, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch + 1, f"{avg_loss:.6f}", current_lr])
            f.flush()

            saved = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "timesteps": T,
                "schedule": "cosine",
                "seed": SEED,
                "betas": betas,
                "alphas": alphas,
                "alpha_bars": alpha_bars,
            }
            torch.save(saved, latest_path)

            # Preserve milestone checkpoints for comparison.
            if epoch + 1 in (50, 75, 100):
                torch.save(saved, OUT_DIR / f"shape_diffusion_epoch_{epoch + 1}.pth")

            print(
                f"Epoch {epoch + 1:03d}/{TARGET_EPOCH} | "
                f"loss={avg_loss:.4f} | lr={current_lr:.1e}",
                flush=True
            )

    print("Latest checkpoint:", latest_path)
    print("Continuation log:", log_path)
    print("Finished at epoch:", TARGET_EPOCH)

if __name__ == "__main__":
    main()
