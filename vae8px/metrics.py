"""The numbers in the note.

`measures` is the latent-bank metric: median later, over the first three
texture windows. `phase_loss`, `horizontal_error`, and `perceptual_patches`
are the photo terms used by the four-arm run and the follow-up.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

_WINDOW = np.outer(np.hanning(64), np.hanning(64))
_LUMA = np.array([0.2126, 0.7152, 0.0722])


def measures(rgb01: np.ndarray) -> dict:
    """One 256 crop, RGB in 0..1. Same bins as analyze_bank.measures."""
    luma = rgb01 @ _LUMA
    gy, gx = np.gradient(luma)
    gradient = float(np.sqrt(np.mean(gx * gx + gy * gy)))
    patches = []
    for y in range(0, 193, 32):
        for x in range(0, 193, 32):
            patch = luma[y : y + 64, x : x + 64]
            patches.append((patch - patch.mean()) * _WINDOW)
    power = np.mean(np.abs(np.fft.fftshift(np.fft.fft2(np.stack(patches)), axes=(-2, -1))) ** 2, axis=0)
    excess = np.exp(np.log(power + 1e-12) - gaussian_filter(np.log(power + 1e-12), 2))
    grid8 = float(np.mean([excess[24, 32], excess[40, 32], excess[32, 24], excess[32, 40]]))
    diag4 = float(np.mean([excess[16, 16], excess[16, 48], excess[48, 16], excess[48, 48]]))
    high = luma - gaussian_filter(luma, 3)
    phase = high.reshape(128, 2, 128, 2).mean((0, 2))
    return dict(
        gradient255=gradient * 255,
        phase2_gray=float(np.std(phase) * 255),
        grid8_excess=grid8,
        grid4_diag_excess=diag4,
    )


def forehead_spectrum(rgb01: np.ndarray) -> dict:
    """Stride-16 spectrum used on the October 12 forehead crop."""
    luma = rgb01 @ _LUMA
    window = np.outer(np.hanning(64), np.hanning(64))
    patches = []
    for y in range(0, luma.shape[0] - 63, 16):
        for x in range(0, luma.shape[1] - 63, 16):
            patch = luma[y : y + 64, x : x + 64]
            patches.append(np.abs(np.fft.fftshift(np.fft.fft2((patch - patch.mean()) * window))) ** 2)
    excess = np.exp(np.log(np.mean(patches, 0) + 1e-9) - gaussian_filter(np.log(np.mean(patches, 0) + 1e-9), 2))
    return dict(
        p8_excess=float(np.mean([excess[24, 32], excess[40, 32], excess[32, 24], excess[32, 40]])),
        p4_diagonal_excess=float(excess[16, 48]),
    )


def phase_loss(error, period=2, window=32):
    # Local phase of the reconstruction error, so a truly periodic target is not punished.
    batch, channels, height, width = error.shape
    assert height % window == 0 and width % window == 0 and window % period == 0
    folded = error.reshape(batch, channels, height // window, window, width // window, window).permute(0, 1, 2, 4, 3, 5)
    means = folded.reshape(
        batch,
        channels,
        height // window,
        width // window,
        window // period,
        period,
        window // period,
        period,
    ).mean((4, 6))
    return (means - means.mean((-2, -1), keepdim=True)).square().mean()


def lattice_error(error):
    return sum(phase_loss(error, period, window) for period, window in ((2, 8), (4, 16), (8, 32))) / 3


def horizontal_error(error):
    # Fourier projection of the paired RGB error onto an 8px horizontal wave. Not image energy.
    length = 64
    coord = torch.arange(length, device=error.device, dtype=torch.float32)
    hann = torch.hann_window(length, periodic=False, device=error.device)
    window = hann[:, None] * hann[None, :]
    phase = 2 * math.pi * coord / 8
    kernels = torch.stack([window * torch.cos(phase)[:, None], window * torch.sin(phase)[:, None]]) / window.sum()
    flat = error.reshape(-1, 1, *error.shape[-2:])
    response = F.conv2d(flat, kernels[:, None], stride=32)
    return response.square().sum(1).mean()


def perceptual_patches(net, prediction, target):
    height, width = prediction.shape[-2:]
    coords = [
        (height // 4 - 256, width // 4 - 256),
        (height // 4 - 256, 3 * width // 4 - 256),
        (3 * height // 4 - 256, width // 4 - 256),
        (3 * height // 4 - 256, 3 * width // 4 - 256),
    ]
    return torch.stack(
        [net(prediction[:, :, y : y + 512, x : x + 512], target[:, :, y : y + 512, x : x + 512]).mean() for y, x in coords]
    ).mean()
