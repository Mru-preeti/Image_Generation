import torch
import torch.nn as nn


# -------------------------
# Time Embedding
# -------------------------
class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):

        device = time.device

        half_dim = self.dim // 2

        embeddings = torch.log(
            torch.tensor(10000.0, device=device)
        ) / (half_dim - 1)

        embeddings = torch.exp(
            torch.arange(
                half_dim,
                device=device
            ) * -embeddings
        )

        embeddings = time[:, None] * embeddings[None, :]

        embeddings = torch.cat(
            (embeddings.sin(), embeddings.cos()),
            dim=-1
        )

        return embeddings


# -------------------------
# Basic convolution block
# -------------------------
class Block(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.norm = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU()

    def forward(self, x):

        return self.activation(
            self.norm(
                self.conv(x)
            )
        )


# -------------------------
# Simple U-Net
# -------------------------
class UNet(nn.Module):

    def __init__(self, time_dim=256):

        super().__init__()

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Encoder
        self.down1 = Block(3, 64)

        self.down2 = Block(64, 128)

        # Bottleneck
        self.bottleneck = Block(128, 256)

        # Decoder
        self.up1 = Block(256 + 128, 128)

        self.up2 = Block(128 + 64, 64)

        # Final output
        self.output = nn.Conv2d(
            64,
            3,
            kernel_size=1
        )

        # Downsampling
        self.pool = nn.MaxPool2d(2)

        # Upsampling
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="nearest"
        )

        # Time projections
        self.time1 = nn.Linear(time_dim, 64)

        self.time2 = nn.Linear(time_dim, 128)

        self.time3 = nn.Linear(time_dim, 256)

    def forward(self, x, t):

        # -------------------------
        # Time embedding
        # -------------------------

        t = self.time_embedding(t)

        # -------------------------
        # Encoder
        # -------------------------

        x1 = self.down1(x)

        t1 = self.time1(t)[:, :, None, None]

        x1 = x1 + t1

        x2 = self.down2(
            self.pool(x1)
        )

        t2 = self.time2(t)[:, :, None, None]

        x2 = x2 + t2

        # -------------------------
        # Bottleneck
        # -------------------------

        x3 = self.bottleneck(
            self.pool(x2)
        )

        t3 = self.time3(t)[:, :, None, None]

        x3 = x3 + t3

        # -------------------------
        # Decoder
        # -------------------------

        x = self.upsample(x3)

        x = torch.cat(
            [x, x2],
            dim=1
        )

        x = self.up1(x)

        x = self.upsample(x)

        x = torch.cat(
            [x, x1],
            dim=1
        )

        x = self.up2(x)

        # -------------------------
        # Predict noise
        # -------------------------

        return self.output(x)