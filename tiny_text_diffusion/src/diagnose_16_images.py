
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT = Path("/content/drive/MyDrive/tiny_text_diffusion")
DATA_DIR = PROJECT / "dataset/synthetic_shapes"
OUT_DIR = PROJECT / "outputs/metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = 50
SEED = 42
DEVICE = torch.device("cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

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

class First16(Dataset):
    def __init__(self):
        with open(DATA_DIR / "labels.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.rows = rows[:16]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img = Image.open(DATA_DIR / self.rows[idx]["filename"]).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1
        return torch.from_numpy(arr).permute(2, 0, 1)

def main():
    ckpt_path = PROJECT / "checkpoints/cosine_continuation/shape_diffusion_cosine_latest.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing reference checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model = TinyUNet().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    alpha_bars = ckpt["alpha_bars"].to(DEVICE)
    loader = DataLoader(First16(), batch_size=16, shuffle=False)

    x0 = next(iter(loader)).to(DEVICE)
    print("Reference checkpoint epoch:", ckpt["epoch"])
    print("Fixed diagnostic images:", len(x0))
    print("Testing noise prediction at selected timesteps:")

    with torch.no_grad():
        for t_value in [1, 5, 10, 20, 35, 49]:
            t = torch.full((len(x0),), t_value, dtype=torch.long)
            noise = torch.randn_like(x0)
            a = alpha_bars[t_value]
            xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
            pred = model(xt, t)

            model_mse = F.mse_loss(pred, noise).item()
            baseline_mse = F.mse_loss(torch.zeros_like(noise), noise).item()
            print(
                f"t={t_value:2d} | model MSE={model_mse:.4f} "
                f"| zero baseline={baseline_mse:.4f}"
            )

    print("Diagnostic completed. No checkpoint was modified.")

if __name__ == "__main__":
    main()
