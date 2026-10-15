"""Where the 8px and 4px components show up inside a feature map.

`phase` is the forehead measurement from the October 12 trace. On a 1024
decode, the crop is pixels [160:400, 352:672]. At coarser feature maps the
same crop is scaled by 1024 / width. The three rows are horizontal, vertical,
and diagonal 2x2 phase.
"""
from __future__ import annotations

import torch


def phase(features):
    stride = 1024 // features.shape[-1]
    crop = features[0, :, 0, 160 // stride : 400 // stride, 352 // stride : 672 // stride].double()
    channels, height, width = crop.shape
    blocks = crop.reshape(channels, height // 2, 2, width // 2, 2).mean((1, 3))
    horizontal = (blocks[:, :, 0] - blocks[:, :, 1]).mean(1) / 2
    vertical = (blocks[:, 0, :] - blocks[:, 1, :]).mean(1) / 2
    diagonal = (blocks[:, 0, 0] + blocks[:, 1, 1] - blocks[:, 0, 1] - blocks[:, 1, 0]) / 4
    std = (crop - crop.mean((-2, -1), keepdim=True)).square().mean().sqrt()
    return torch.stack([horizontal, vertical, diagonal]), std


def stat(features) -> dict:
    component, std = phase(features)
    axis = component[:2].square().sum(0).mean().sqrt()
    diagonal = component[2].square().mean().sqrt()
    return dict(
        spatial_std=float(std),
        axis_phase_rms=float(axis),
        diagonal_phase_rms=float(diagonal),
        axis_over_std=float(axis / std),
        diagonal_over_std=float(diagonal / std),
        axis_amplitude=float(axis),
    )


def axis8_notch(features, width=0.0075):
    """Narrow Gaussian notch at the 8px axial frequencies, in feature space."""
    stride = 1024 / features.shape[-1]
    height, width_px = features.shape[-2:]
    freq_y = torch.fft.fftfreq(height, device=features.device)[:, None]
    freq_x = torch.fft.fftfreq(width_px, device=features.device)[None, :]
    mask = torch.ones((height, width_px), device=features.device)
    centers = {
        (round((y * stride + 0.5) % 1 - 0.5, 6), round((x * stride + 0.5) % 1 - 0.5, 6))
        for y, x in ((0.125, 0), (-0.125, 0), (0, 0.125), (0, -0.125))
    }
    for center_y, center_x in centers:
        dy = torch.remainder(freq_y - center_y + 0.5, 1) - 0.5
        dx = torch.remainder(freq_x - center_x + 0.5, 1) - 0.5
        mask = mask * (1 - torch.exp(-(dy.square() + dx.square()) / (2 * (width * stride) ** 2)))
    result = torch.empty_like(features)
    for start in range(0, features.shape[1], 64):
        block = features[:, start : start + 64]
        result[:, start : start + 64] = torch.fft.ifft2(torch.fft.fft2(block) * mask).real
    return result
