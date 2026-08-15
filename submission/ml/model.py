"""
Compact U-Net that predicts a coarse match heatmap.

Input : (N, 2, 256, 256)  search image + tiled reference context
Output: (N, 1, 64, 64)    raw logits -> sigmoid = match score heatmap

The decoder stops at 64x64 (the heatmap resolution); encoders provide global
context so the network can disambiguate repeated-pattern matches, while the
small final field of view keeps training fast on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HeatmapUNet(nn.Module):
    def __init__(self, in_ch: int = 2, base: int = 32, out_ch: int = 1):
        super().__init__()
        self.enc1 = _ConvBlock(in_ch, base)          # 256
        self.pool1 = nn.MaxPool2d(2)                 # 128
        self.enc2 = _ConvBlock(base, base * 2)       # 128
        self.pool2 = nn.MaxPool2d(2)                 # 64
        self.enc3 = _ConvBlock(base * 2, base * 4)   # 64
        self.pool3 = nn.MaxPool2d(2)                 # 32
        self.enc4 = _ConvBlock(base * 4, base * 8)   # 32
        self.pool4 = nn.MaxPool2d(2)                 # 16
        self.bottleneck = _ConvBlock(base * 8, base * 8)

        self.up4 = nn.ConvTranspose2d(base * 8, base * 8, 2, stride=2)   # -> 32
        self.dec4 = _ConvBlock(base * 16, base * 4)                      # 32
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)   # -> 64
        self.dec3 = _ConvBlock(base * 8, base * 2)                       # 64
        self.head = nn.Conv2d(base * 2, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d = self.dec3(torch.cat([self.up3(d), e3], dim=1))
        return self.head(d)


def soft_argmax(logits: torch.Tensor, temperature: float = 0.1):
    """Differentiable argmax over the 64x64 heatmap -> (x, y) in cell units."""
    B = logits.shape[0]
    logits_flat = logits.view(B, -1)
    p = torch.softmax(logits_flat / temperature, dim=1)
    size = logits.shape[-1]
    yy, xx = torch.meshgrid(
        torch.arange(size, device=logits.device),
        torch.arange(size, device=logits.device),
        indexing="ij",
    )
    x = torch.sum(p * xx.reshape(-1), dim=1)
    y = torch.sum(p * yy.reshape(-1), dim=1)
    return torch.stack([x, y], dim=1)


def heatmap_loss(pred_logits, target_gauss, gt_cells, temperature=0.1,
                 w_mse=5.0, w_sa=1.0, w_peak=0.5):
    """Combined loss:
      * peak-weighted MSE against the Gaussian target,
      * SmoothL1 on the soft-argmax (directly optimizes centre position),
      * a small penalty so the map is forced to contain a peak.
    """
    heat = torch.sigmoid(pred_logits)                        # B,1,H,H
    target = target_gauss                                    # B,1,H,H
    weight = 1.0 + 5.0 * target
    mse = ((heat - target) ** 2 * weight).mean()

    sa = soft_argmax(pred_logits, temperature)
    sa_loss = nn.functional.smooth_l1_loss(sa, gt_cells, beta=1.0)

    peak = (1.0 - heat.flatten(1).max(dim=1).values).mean()
    return w_mse * mse + w_sa * sa_loss + w_peak * peak, {
        "mse": float(mse.item()),
        "sa": float(sa_loss.item()),
        "peak": float(peak.item()),
    }
