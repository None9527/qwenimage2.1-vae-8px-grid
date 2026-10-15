"""Fake textures shown to the unconditional critic.

Arms C and A learn to rank a real crop above a periodic copy and a blurred
copy of itself. These two functions build those copies.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def periodic(image, period=None, amplitude=None):
    count, _channels, height, width = image.shape
    period = random.choice([2, 4, 8]) if period is None else period
    phase = torch.rand(count, 1, 1, 1, device=image.device) * 2 * torch.pi
    if period == 2:
        phase = torch.randint(0, 2, (count, 1, 1, 1), device=image.device) * torch.pi
    xx = torch.arange(width, device=image.device)[None, None, None, :]
    yy = torch.arange(height, device=image.device)[None, None, :, None]
    vertical = torch.cos(yy * 2 * torch.pi / period + phase)
    horizontal = torch.cos(xx * 2 * torch.pi / period + phase)
    patterns = torch.stack(
        [vertical.expand(count, 1, height, width), horizontal.expand(count, 1, height, width), vertical * horizontal],
        1,
    )
    kinds = torch.randint(0, 3, (count,), device=image.device)
    pattern = patterns[torch.arange(count, device=image.device), kinds]
    pattern = pattern / pattern.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
    amp = (torch.rand(count, 1, 1, 1, device=image.device) * 3 + 1) if amplitude is None else amplitude
    return (image + pattern * amp / 127.5).clamp(-1, 1)


def blurred(image):
    return image * 0.5 + F.avg_pool2d(F.pad(image, (1, 1, 1, 1), mode="reflect"), 3, 1) * 0.5
